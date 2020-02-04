# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin
from pyfr.util import memoize, subclass_where
from pyfr.shapes import BaseShape
from pyfr.writers.vtk import BaseShapeSubDiv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

class CoprocessPlugin(BasePlugin):
    name = 'coprocess'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')
        self.divisor = self.cfg.getint('solver', 'order') + 1
        self.dtype = np.float32

        self.planes = {}
        for name, value in self.cfg.items(self.cfgsect).items():
            if name.startswith('plane') and name.endswith('origin'):
                tmp = name.split('_')
                nname = '_'.join(tmp[:-1])
                origin = [float(x) for x in value.split(',')]
                value = self.cfg.get(self.cfgsect, '%s_normal'%nname)
                normal = [float(x) for x in value.split(',')]
                self.planes[nname] = (origin, normal)

        intg = args[0]
        self.mesh = intg.system.mesh
        self.elementscls = intg.system.elementscls
        self._create_mb()

    def _create_mb(self):

        mesh = self.mesh
        elementscls = self.elementscls

        # Create the vtkMultiBlockDataSet
        mesh_inf = mesh.array_info('spt')
        
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(len(mesh_inf))

        iblk = 0
        for mk, (name, shape) in mesh_inf.items():
            
            msh = mesh[mk].astype(self.dtype)

            # Dimensions
            nspts, neles, ndims = shape

            # Sub divison points inside of a standard element
            svpts = self._get_std_ele(name, nspts)
            nsvpts = len(svpts)

            # Generate the operator matrices
            mesh_vtu_op = self._get_mesh_op(name, nspts, svpts)
            
            # Calculate node locations of VTU elements
            vpts = mesh_vtu_op @ msh.reshape(nspts, -1)
            vpts = vpts.reshape(nsvpts, -1, self.ndims)

            # Pre-process the solution
            soln_fields = list(elementscls.privarmap[ndims])

            # Append dummy z dimension for points in 2D
            if ndims == 2:
                vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')
            
            # Perform the sub division
            subdvcls = subclass_where(BaseShapeSubDiv, name=name)
            nodes = subdvcls.subnodes(self.divisor)

            # Prepare VTU cell arrays
            vtu_con = np.tile(nodes, (neles, 1))
            vtu_con += (np.arange(neles)*nsvpts)[:, None]
            grid = vtk.vtkUnstructuredGrid()

            npts_total = vpts.shape[0] * vpts.shape[1]
            vtk_coords = numpy_to_vtk(vpts.swapaxes(0, 1).reshape(-1, 3), deep=True)
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(vtk_coords)
            grid.SetPoints(vtk_points)

            subcelloff = subdvcls.subcelloffs(self.divisor)
            nsubcell = len(subcelloff)
            subcelloff = np.append(0, subcelloff)
            subcelltypes = subdvcls.subcelltypes(self.divisor)

            n_cells = 0
            all_conn = []
            all_types = []
            all_offsets = []
            for typ in np.unique(subcelltypes):
                idx = np.nonzero(subcelltypes == typ)[0]
                tmp = [vtu_con[:, subcelloff[i]:subcelloff[i+1]] for i in idx]
                conn = np.vstack(tmp)
                n, m = conn.shape
                conn = np.hstack((m * np.ones((n, 1), dtype=np.int),
                                  conn)).ravel()
                types = typ * np.ones(n, dtype=int)
                offsets = m*np.arange(n) + n_cells
                n_cells += n

                all_conn.append(conn)
                all_types.append(types)
                all_offsets.append(offsets)

            all_conn = np.concatenate(all_conn)
            all_types = np.concatenate(all_types)
            all_offsets = np.concatenate(all_offsets)


            vtk_cells = vtk.vtkCellArray()
            vtk_cells.SetCells(n_cells, numpy_to_vtk(all_conn,
                                                      deep=True,
                                                      array_type=vtk.vtkIdTypeArray().GetDataType()))
            vtk_cell_types = numpy_to_vtk(all_types,
                                          deep=True,
                                          array_type=vtk.vtkUnsignedCharArray().GetDataType())
            vtk_cell_locations = numpy_to_vtk(all_offsets,
                                              deep=True,
                                              array_type=vtk.vtkIdTypeArray().GetDataType())

            grid.SetCells(vtk_cell_types, vtk_cell_locations, vtk_cells)

            mb.SetBlock(iblk, grid)
            mb.GetMetaData(iblk).Set(vtk.vtkCompositeDataSet.NAME(), name)

            nodedata = grid.GetPointData()

            for name in soln_fields:
                tmp = np.zeros(npts_total, dtype=self.dtype)
                vtk_arr = numpy_to_vtk(tmp, deep=True)
                vtk_arr.SetName(name)
                nodedata.AddArray(vtk_arr)

        self._mb = mb

    def _update_sol(self, soln):

        mesh_inf = self.mesh.array_info('spt')

        elementscls = self.elementscls
        nblks = self._mb.GetNumberOfBlocks()

        iblk = 0
        for mk, (name, shape) in mesh_inf.items():
            grid = self._mb.GetBlock(iblk)

            # Dimensions
            nspts, neles, ndims = shape
            svpts = self._get_std_ele(name, nspts)
            nsvpts = len(svpts)

            soln = np.array(elementscls.con_to_pri(soln, self.cfg)).swapaxes(0, 1)
            
            # Interpolate the solution to the vis points
            soln_vtu_op = self._get_soln_op(name, nspts, svpts)
            vsoln = soln_vtu_op @ soln.reshape(len(soln), -1)
            vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)
            
            nodedata = grid.GetPointData()

            for idx, arr in enumerate(vsoln):
                vtk_arr = nodedata.GetArray(idx)
                tmp = vtk_to_numpy(vtk_arr)
                tmp[:] = arr.T.ravel()

    @memoize
    def _get_shape(self, name, nspts):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, self.cfg)

    @memoize
    def _get_std_ele(self, name, nspts):
        return self._get_shape(name, nspts).std_ele(self.divisor)

    @memoize
    def _get_mesh_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    @memoize
    def _get_soln_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

    def _cut_plane(self, origin, normal):

        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        planeCut = vtk.vtkCutter()
        planeCut.SetCutFunction(plane)
        planeCut.GenerateCutScalarsOn()

        res = []
        for iblk in range(self._mb.GetNumberOfBlocks()):
            planeCut.SetInputData(self._mb.GetBlock(0))
            planeCut.Update()
            res.append(planeCut.GetOutput())

        return res

    def __call__(self, intg):

        if intg.nacptsteps % self.nsteps == 0:
            print('coprocess iter %d'%intg.nacptsteps)

            # Solution
            soln = intg.soln
            assert(len(soln) == 1)
            soln = soln[0].swapaxes(0, 1)

            self._update_sol(soln)

            for name, (origin, normal) in self.planes.items():
                tmp = self._cut_plane(origin, normal)

                for idx, grid in enumerate(tmp):
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetCompressorTypeToNone()
                    writer.SetDataModeToBinary()
                    writer.SetInputData(grid)
                    writer.SetFileName('%s_%d_%d.vtp'%(name,idx,intg.nacptsteps))
                    writer.Write()

            print('done')