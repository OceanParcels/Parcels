"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np
import netCDF4
from datetime import timedelta as delta
from parcels.tools.loggers import logger

import os 
from tempfile import gettempdir
import psutil
from parcels.tools.error import ErrorCode

try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['ParticleFile']


class ParticleFile(object):
    """Initialise netCDF4.Dataset for trajectory output.

    The output follows the format outlined in the Discrete Sampling Geometries
    section of the CF-conventions:
    http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

    The current implementation is based on the NCEI template:
    http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

    Developer note: We cannot use xray.Dataset here, since it does not yet allow
    incremental writes to disk: https://github.com/pydata/xarray/issues/199

    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param chunksizes: 2d vector of sizes for NetCDF chunking. Lower values means smaller files, but also slower IO.
                       See e.g. https://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes
    """

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, chunksizes=None):

        self.name = name
        self.chunksizes = [max([len(particleset), 1]), 1] if chunksizes is None else chunksizes
        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttraj = 0  # id of last particle written
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
        self.particleset = particleset
        
        self.npy_path = os.path.join(gettempdir(), "parcels-%s" % os.getuid(), "out")
        if os.path.exists(self.npy_path):
            os.system("rm -rf "+ self.npy_path)
            print "Existing temporary output folder ('"+self.npy_path+"') from previous runs (probably aborted) was deleted"

    def open_dataset(self):
        extension = os.path.splitext(str(self.name))[1]
        fname = self.name if extension in ['.nc', '.nc4'] else "%s.nc" % self.name
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", None)
        self.dataset.createDimension("traj", None)
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
        
        
        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i4", coords, fill_value=-2147483647, chunksizes=self.chunksizes)
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"
        
        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if self.particleset.time_origin.calendar is None:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(self.particleset.time_origin)
            self.time.calendar = self.particleset.time_origin.calendar
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        self.user_vars = []
        self.user_vars_once = []
        """
        :user_vars: list of additional user defined particle variables to write for all particles and all times
        :user_vars_once: list of additional user defined particle variables to write for all particles only once at initial time.
        """

        for v in self.particleset.ptype.variables:
            if v.name in ['time', 'lat', 'lon', 'depth', 'z', 'id']:
                continue
            if v.to_write:
                if v.to_write is True:
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", coords, fill_value=np.nan, chunksizes=self.chunksizes))
                    self.user_vars += [v.name]
                elif v.to_write == 'once':
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", "traj", fill_value=np.nan, chunksizes=[self.chunksizes[0]]))
                    self.user_vars_once += [v.name]
                getattr(self, v.name).long_name = ""
                getattr(self, v.name).standard_name = v.name
                getattr(self, v.name).units = "unknown"

        self.idx = np.empty(shape=0)

    def __del__(self):
        if self.dataset:
            self.convert_npy()
            self.dataset.close()

    def sync(self):
        """Write all buffered data to disk"""
        self.dataset.sync()

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`
        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.open_dataset()
        setattr(self.dataset, name, message)
            
    def write(self, pset, time, sync=True, deleted_only=False):
        """Write :class:`parcels.particleset.ParticleSet` 
        All data from one time step is saved to one NPY-file using a python 
        dictionary. The data is saved in the folder 'out'.

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param sync: Optional argument whether to write data to disk immediately. Default is True

        """
        if self.dataset is None:
            self.open_dataset()
        if isinstance(time, delta):
            time = time.total_seconds()
        if self.lasttime_written != time and \
           (self.write_ondelete is False or deleted_only is True):
            if pset.size > 0:

                first_write = [p for p in pset if (p.fileid < 0 or len(self.idx) == 0) and (p.dt*p.time <= p.dt*time or np.isnan(p.dt))]  # len(self.idx)==0 in case pset is written to new ParticleFile
                for p in first_write:
                    p.fileid = self.lasttraj
                    self.lasttraj += 1

                self.idx = np.append(self.idx, np.zeros(len(first_write)))

                size = len(pset)
                
                # dictionary for temporary hold data                
                tmp = {}
                tmp["ids"], tmp["time"], tmp["lat"], tmp["lon"], tmp["z"] =\
                        map(lambda x: np.zeros(x), [size,size,size,size,size])
                
                for var in self.user_vars:
                    tmp[var] = np.zeros(size)
                
                for key in tmp.keys():
                    tmp[key][:] = np.nan 
                
                i = 0
                for p in pset:

                    if p.dt*p.time <= p.dt*time: 
                        tmp["ids"][i] = p.id
                        tmp["time"][i] = time
                        tmp["lat"][i] = p.lat
                        tmp["lon"][i] = p.lon
                        tmp["z"][i]   = p.depth
                        for var in self.user_vars:
                            tmp[var][i] = getattr(p, var)
                        if p.state != ErrorCode.Delete and not np.allclose(p.time, time):
                            logger.warning_once('time argument in pfile.write() is %g, but a particle has time %g.' % (time, p.time))
                        i += 1
                
                if not os.path.exists(self.npy_path):
                    os.mkdir(self.npy_path)
                
                save_ind = np.isfinite(tmp["ids"])
                for key in tmp.keys():
                    tmp[key] = tmp[key][save_ind]

                np.save(os.path.join(self.npy_path,str(time)),tmp)
                
                for p in first_write:
                    for var in self.user_vars_once:
                        getattr(self, var)[p.fileid] = getattr(p, var)
            else:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)

            if not deleted_only:
                self.idx += 1
                self.lasttime_written = time

        if sync:
            self.sync()
            
    def convert_npy(self,delete_tmp = True, batch_processing=False, batch_size = 2**30):
        """Writes outputs from NPY-files to ParticleFile instance
        
        :param delete_tmp: If True the tmp output folder for the NPY-files is deleted. Default: True
        
        :param batch_processing: If True batch processing is applied. Batches 
        of files are loaded and then written to the output netcdf-file. This
        option is still work in progress. Default:True
        
        :param batch_size: size of one batch in bytes. Default: 2**30 (1 Gbyte).
        
        :type delete_tmp: boolean
        :type batch_processing: boolean
        :type batch_size: int
        """
        
        def get_info(file_list):
            """get shape and variable names of the temporary output files
            """
            
            # infer array size id from the highest id in NPY file from last time step
            data_dict  = np.load(os.path.join(self.npy_path,str(file_list[-1])+".npy")).item()

            n_id = int(max(data_dict["ids"])+1)
            n_time = len(file_list)
            var_names = data_dict.keys()
            
            return n_id,n_time,var_names
        
        def init_merge_dict(time_steps):
            """initalize a merging directory"""
            
            # dictionary for merging
            merge_dict = {}
            for var in self.var_names:
                merge_dict[var] = np.zeros((self.n_id,time_steps))
                
                if var!="ids":
                    merge_dict[var][:] = np.nan
                else:
                    merge_dict[var][:] = self.id._FillValue
            return merge_dict
        
        def get_available_memory():
            """return available memory in bytes"""
            memory_info = psutil.virtual_memory()
            available_memory = memory_info.available
            return available_memory
        
        def read(file_list,time_steps):
            """Read NPY-files using a loop over all files and return one array 
            for each variable. 
            
            :param file_list: List that  contains all file names in the output 
            directory
            :param id_fill_value: FillValue for the IDs as used in the output
            netcdf
            
            :type file_list: list
            :type id_fill_value: int
                
            :return: Python dictionary with the data that was read from the 
            NPY-files
            """
            
            merge_dict = init_merge_dict(time_steps)
            # initiated indeces for time axis
            time_index = np.zeros(self.n_id,dtype=int)
            
            # loop over all files
            for i in range(time_steps):
                
                data_dict = np.load(os.path.join(self.npy_path,str(file_list[i])+".npy")).item()
                
                # don't convert to netdcf if all values are nan for a time step
                if np.isnan(data_dict["ids"]).all():
                    for key in merge_dict.keys():
                        merge_dict[key] = merge_dict[key][:,:-1]
                    continue
                
                # get ids that going to be filled
                id_ind =  np.array(data_dict["ids"],dtype=int)
                
                # get the corresponding time indeces
                t_ind = time_index[id_ind]
                
                # write into merge array
                for key in self.var_names:
                    merge_dict[key][id_ind,t_ind] = data_dict[key]
               
                # new time index for ids that where filled with values
                time_index[id_ind]  = time_index[id_ind]  + 1
            
            # remove rows that are completely filled with nan values
            out_dict = {}
            for var in self.var_names:
                out_dict[var] = merge_dict[var][~np.isnan(merge_dict["lat"]).all(axis=1)]
            
            return out_dict

        # sort list of NPY-files
        time_list = os.listdir(self.npy_path)
        time_list = [x[:-4] for x in time_list]
        time_list = map(float,time_list)
        time_list.sort()
        
        self.n_id,self.n_time,self.var_names = get_info(time_list)
        
        if batch_processing:
            print "=============convert NPY-files to NetCDF-file==============="
        
            available_memory = get_available_memory()
            print "Available memory: '" + str(available_memory/float(2**20)) + "' Mbytes"
            self.batch_size = batch_size
            
            # estimate the total size in bytes for merging array
            self.memory_estimate_total = self.n_id * self.n_time  * len(self.var_names) *  8
            self.memory_per_file = self.memory_estimate_total/len(time_list)
            print "Estimated memory needed: '" +str(float(self.memory_estimate_total)/2**20)+"' Mbytes" 
            print "Estimated memory per file needed: '" +str(float(self.memory_per_file)/2**20)+"' Mbytes" 
            
            if self.memory_per_file>available_memory*0.9:
                raise MemoryError("Too little available memory is available to load even one tempory output file. With 10% safety margin.")
            
            if 0.7*available_memory<self.batch_size:
                self.batch_size = 0.7 * available_memory 
                print "Too little available  memory available in ParticleFile.conversion_npy() for batch_size:"+ str(batch_size/2**20) +" Mbytes."
                print "Setting batch_size to 70% of available  memory:"+str(float(self.batch_size)/2**20) + " Mbytes."                             
                        
            if batch_size<self.memory_per_file:
                self.batch_size = self.memory_per_file
                print "'batch_size' size too little.'batch_size' has to be at least as big as the memory needed per file."
                print "Set batch the memory that is needed per file: "+ str(self.batch_size/2**20) + "Mbytes."
                
            print "Convert NPY-files using batch processing in batches of max. size '"+str(float(self.batch_size)/2**20)+"' Mbytes." 
            
            # divide the data set into chunks
            n_reading_loops = int(self.memory_estimate_total//self.batch_size)
            if n_reading_loops == 0:
                n_reading_loops = 1
            
            files_per_loop = len(time_list)//n_reading_loops
            
            if files_per_loop==0:
                raise ValueError("'files_per_loop' can not be equal to 0. Some bug might exist in ParticleFile.convert_npy().")
            
            time_list_splitted = [time_list[i:i + files_per_loop] for i in range(0, len(time_list), files_per_loop)]
            
            print "Convert data using " +str(len(time_list_splitted))+" batch(es)." 
            
            # last time index that was filled
            last_filled = 0
            
            # reading loop
            for time_list_loop in time_list_splitted:
                n_time_step = len(time_list_loop)
                data_dict = read(time_list_loop,n_time_step)
                
                for var in self.var_names:
                    if var !="ids":
                        getattr(self, var)[:,last_filled:last_filled+n_time_step] = data_dict[var]
                    else:
                        getattr(self, "id")[:,last_filled:last_filled+n_time_step] = data_dict[var]      
                
                last_filled += n_time_step
        
        else:
            data_dict = read(time_list,len(time_list))
            for var in self.var_names:
                if var !="ids":
                    getattr(self, var)[:,:] = data_dict[var]
                else:
                    getattr(self, "id")[:,:] = data_dict[var]  
        
        if os.path.exists(self.npy_path) and delete_tmp:
            print "Remove folder '"+self.npy_path+"' after conversion of NPY-files to NetCDF file '"+str(self.name)+"'." 
            os.system("rm -rf "+self.npy_path)
