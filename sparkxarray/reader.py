""" Interface for Data Ingestion.
"""
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np
import pandas as pd 
import xarray as xr
import itertools
from glob import glob
from pyspark.sql import SparkSession

def ncread(sc, paths, mode='single', **kwargs):
    """Calls sparkxarray netcdf read function based on the mode parameter.

    ============ ==============================
    Mode          Reading Function
    ------------ ------------------------------
    single       : read_nc_single
    multi        : read_nc_multi
    Anything else: Throw an exception
    ============= ==============================

    Parameters
    ----------

    sc       :  sparkContext object

    paths    :  str or sequence
                Either a string glob in the form "path/to/my/files/*.nc" or an explicit
                list of files to open

    mode     : str
               'single' for a single file
               'multi' for multiple files
               'hdfs' for reading multiple files as binary from HDFS, then using them as in-memory NetCDF files, and then partitioning with groupBy

    **kwargs : dict
               partitioning options to be passed on to the actual read function.
            
    
    """

    if 'partitions' not in kwargs:
        kwargs['partitions'] = None

    if 'partition_on' not in kwargs:
        kwargs['partition_on'] = ['time']

    error_msg = ("You specified a mode that is not implemented.")

    if (mode == 'single'):
        return read_nc_single(sc, paths, **kwargs)

    elif (mode == 'multi'):
        return read_nc_multi(sc, paths, **kwargs)
    elif (mode == 'hdfs'):
        return read_nc_hdfs(sc, paths, **kwargs)
    else:
        raise NotImplementedError(error_msg)

        
def read_nc_single(sc, paths, **kwargs):
    """ Read a single netCDF file

    Parameters
    -----------
    sc       :  sparkContext object

    paths    :  str
                an explicit filename to open
    

    **kwargs : dict
               Additional arguments for partitioning 

    """
    partition_on = kwargs.get('partition_on')
    partitions = kwargs.get('partitions')

    dset = xr.open_dataset(paths)

    # D = {'dim_1': dim_1_size, 'dim_2': dim_2_size, ...}
    D ={dset[dimension].name:dset[dimension].size for dimension in partition_on}
    
    # dim_sizes = [range(dim_1_size), range(dim_2_size), range(...)]
    dim_ranges = [range(dim_size) for dim_size in D.values()]
    

    dim_cartesian_product_indices = [element for element in itertools.product(*dim_ranges)]

    # create a list of dictionaries for  positional indexing
    positional_indices = [dict(zip(partition_on, ij)) for ij in dim_cartesian_product_indices]

    if not partitions:
        partitions = len(dim_cartesian_product_indices) / 50

    if partitions > len(dim_cartesian_product_indices):
        partitions = len(dim_cartesian_product_indices)

    
    # Create an RDD
    rdd = sc.parallelize(positional_indices, partitions).map(lambda x: readone_slice(dset, x))

    return rdd


def readone_slice(dset, positional_indices):
    """Read a slice from an xarray.Dataset.

    Parameters
    ----------

    dset                : file_object
                         xarray.Dataset object
    positional_indices  : dict
                          dict containing positional indices for each dimension
                          e.g. {'lat': 0, 'lon': 0}

    Returns
    ---------
    chunk               : xarray.Dataset
                         a subset of the Xarray Dataset

    """

    # Change the positional indices into slice objects
    # e.g {'lat': 0, 'lon': 0} ---> {'lat': slice(0, 1, None),  'lon': slice(0, 1, None)}
    positional_slices = {dim: slice(positional_indices[dim], positional_indices[dim]+1) 
                                                         for dim in positional_indices}

    # Read a slice for the given positional_slices
    chunk = dset[positional_slices]
    return chunk


def read_nc_multi(sc, paths, **kwargs):
    """ Read multiple netCDF files

    Parameters
    -----------
    sc       :  sparkContext object

    paths    :  str or sequence
                Either a string glob in the form "path/to/my/files/*.nc" or an explicit
                list of files to open

    **kwargs : dict
               Additional arguments for partitioning 

    """

    partition_on = kwargs.get('partition_on')
    partitions = kwargs.get('partitions')

    dset = xr.open_mfdataset(paths, autoclose=True)

    # D = {'dim_1': dim_1_size, 'dim_2': dim_2_size, ...}
    D ={dset[dimension].name:dset[dimension].size for dimension in partition_on}
    
    # dim_sizes = [range(dim_1_size), range(dim_2_size), range(...)]
    dim_ranges = [range(dim_size) for dim_size in D.values()]
    

    dim_cartesian_product_indices = [element for element in itertools.product(*dim_ranges)]

    # create a list of dictionaries for  positional indexing
    positional_indices = [dict(zip(partition_on, ij)) for ij in dim_cartesian_product_indices]

    if not partitions:
        partitions = len(dim_cartesian_product_indices) / 50

    if partitions > len(dim_cartesian_product_indices):
        partitions = len(dim_cartesian_product_indices)

    
    # Create an RDD
    rdd = sc.parallelize(positional_indices, partitions).map(lambda x: readone_slice(dset, x))

    return rdd

def read_nc_hdfs(sc, paths, **kwargs):
    """ Read one or more netCDF files from DHFS

    Parameters
    -----------
    sc       :  sparkContext object

    paths    :  str or sequence
                Either a string glob in the form "path/to/my/files/*.nc" or an explicit
                list of files to open

    **kwargs : dict
               Additional arguments for partitioning 

    """
    partition_on = kwargs.get('partition_on')
    partitions = kwargs.get('partitions')
    merge = kwargs.get('merge')

    print("Running spark-xarray in HDFS mode")

    binrdd = sc.binaryFiles(paths)

    def nc_load(data):
        name,nc_bytes = data
        dset = xr.open_dataset(nc_bytes)
        return dset

    datardd = binrdd.map(nc_load)
    if not partition_on:
        return datardd

    def dict_to_key(x):
        if len(x) == 1:
            return next (iter (x.values()))
        else:
            return tuple(x.values())


    def nc_partition(dset):       
        # D = {'dim_1': dim_1_size, 'dim_2': dim_2_size, ...}
        D ={dset[dimension].name:dset[dimension].size for dimension in partition_on}       
        # dim_sizes = [range(dim_1_size), range(dim_2_size), range(...)]
        dim_ranges = [range(dim_size) for dim_size in D.values()]
        dim_cartesian_product_indices = [element for element in itertools.product(*dim_ranges)]
        # create a list of dictionaries for  positional indexing
        positional_indices = [dict(zip(partition_on, ij)) for ij in dim_cartesian_product_indices]
        return map(lambda x: (dict_to_key(x), readone_slice(dset, x).load()), positional_indices)

    if merge:
        rdd = datardd.flatMap(nc_partition).reduceByKey(lambda a, b: a.merge(b, inplace=True))
    else:
        rdd = datardd.flatMap(nc_partition)
    if not partitions:
        return rdd

    return rdd.repartition(partitions)
