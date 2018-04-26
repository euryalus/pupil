'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
# cython: profile=False
from detector cimport *
from methods import  normalize
from numpy.math cimport PI
import numpy as np
from cython.operator cimport dereference as deref


cdef extern from 'singleeyefitter/mathHelper.h' namespace 'singleeyefitter::math':

    Matrix21d cart2sph( Matrix31d& m )

cdef inline convertTo2DCppResult(pupil_datum, shared_ptr[Detector2DResult] ptr):

    deref(ptr).confidence = pupil_datum['confidence']
    deref(ptr).ellipse.center[0] = pupil_datum['ellipse']['center'][0]
    deref(ptr).ellipse.center[1] = pupil_datum['ellipse']['center'][1]
    deref(ptr).ellipse.minor_radius = pupil_datum['ellipse']['axes'][0]/2.0
    deref(ptr).ellipse.major_radius = pupil_datum['ellipse']['axes'][1]/2.0
    deref(ptr).ellipse.angle = (pupil_datum['ellipse']['angle']+90)*PI/180.0
    deref(ptr).timestamp = pupil_datum['timestamp']
    deref(ptr).image_height = pupil_datum['height']
    deref(ptr).image_width = pupil_datum['width']

    deref(ptr).final_edges.clear()
    cdef Point_[int] temp
    for edge in pupil_datum['final_edges']:
        temp.x = int(edge[0])
        temp.y = int(edge[1])
        deref(ptr).final_edges.push_back(temp)

    deref(ptr).current_roi.x = pupil_datum['ROI'][0]
    deref(ptr).current_roi.y = pupil_datum['ROI'][1]
    deref(ptr).current_roi.width = pupil_datum['ROI'][2]
    deref(ptr).current_roi.height = pupil_datum['ROI'][3]

cdef inline convertTo2DPythonResult( Detector2DResult& result, object frame, object roi, edges):

    ellipse = {}
    ellipse['center'] = (result.ellipse.center[0],result.ellipse.center[1])
    ellipse['axes'] =  (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0)
    ellipse['angle'] = result.ellipse.angle * 180.0 / PI - 90.0

    py_result = {}
    py_result['topic'] = 'pupil'
    py_result['confidence'] = result.confidence
    py_result['ellipse'] = ellipse
    py_result['diameter'] = max(ellipse['axes'])

    norm_center = normalize( ellipse['center'] , (frame.width, frame.height),flip_y=True)
    py_result['norm_pos'] = norm_center
    py_result['timestamp'] = frame.timestamp
    py_result['method'] = '2d c++'

    py_result['width'] = result.image_width
    py_result['height'] = result.image_height
    py_result['ROI']= [result.current_roi.x, result.current_roi.y, result.current_roi.width, result.current_roi.height]

    if edges:
        py_result['final_edges'] = getFinalEdges2D(result)
        py_result['raw_edges'] = getRawEdges2D(result)



    return py_result

cdef inline convertTo3DPythonRefractionResult( Detector3DResultRefraction& result):

    py_result = {}

    py_result['resFit'] = result.resFit
    py_result['initial_center'] = result.initial_center
    py_result['initial_center'][1] *= -1
    py_result['optimized_center'] = result.optimized_center
    py_result['optimized_center'][1] *= -1
    py_result['cost'] = result.cost
    py_result['number_of_pupils'] = result.number_of_pupils
    #py_result['par_history'] = np.array(result.par_history)
    #py_result['pupil_type_history'] = np.array(result.pupil_type_history)
    #py_result['cost_history'] = np.array(result.cost_history)
    #py_result['residual_histogram'] = np.array(result.residual_histogram)
    #py_result['mean_residual'] = result.mean_residual
    #py_result['std_residual'] = result.std_residual

    #py_result['edges'] = {}
    #py_result['circles'] = {}
    #py_result['ellipses'] = {}

    #for i in range(py_result['number_of_pupils']):
    #
    #    py_result['edges'][i] = result.edge_map[i]
    #
    #    py_result['circles'][i] =  [result.circles[i].center[0],
    #                                result.circles[i].center[1],
    #                                result.circles[i].center[2],
    #                                result.circles[i].normal[0],
    #                                result.circles[i].normal[1],
    #                                result.circles[i].normal[2],
    #                                result.circles[i].radius]
    #
    #    py_result['ellipses'][i] = [result.ellipses[i].center[0],
    #                                result.ellipses[i].center[1],
    #                                result.ellipses[i].major_radius,
    #                                result.ellipses[i].minor_radius,
    #                                result.ellipses[i].angle]

    return py_result

cdef inline convertTo3DPythonResult(Detector3DResult& result, object frame):

    #use negative z-coordinates to get from left-handed to right-handed coordinate system
    py_result = {}
    py_result['topic'] = 'pupil'

    py_result['cost'] = result.cost

    circle = {}
    circle['center'] =  (result.circle.center[0],-result.circle.center[1], result.circle.center[2])
    circle['normal'] =  (result.circle.normal[0],-result.circle.normal[1], result.circle.normal[2])
    circle['radius'] =  result.circle.radius
    py_result['circle_3d'] = circle

    py_result['confidence'] = result.confidence
    py_result['timestamp'] = frame.timestamp
    py_result['diameter_3d'] = result.circle.radius * 2.0

    ellipse = {}
    ellipse['center'] = (result.ellipse.center[0] + frame.width / 2.0 ,frame.height / 2.0  -  result.ellipse.center[1])
    ellipse['axes'] =  (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0)
    ellipse['angle'] = - (result.ellipse.angle * 180.0 / PI - 90.0)
    py_result['ellipse'] = ellipse
    norm_center = normalize( ellipse['center'] , (frame.width, frame.height),flip_y=True)
    py_result['norm_pos'] = norm_center
    py_result['diameter'] = max(ellipse['axes'])

    sphere = {}
    sphere['center'] =  (result.sphere.center[0],-result.sphere.center[1], result.sphere.center[2])
    sphere['radius'] =  result.sphere.radius
    py_result['sphere'] = sphere

    if str(result.projectedSphere.center[0]) == 'nan':
        projectedSphere = {'axes': (0,0), 'angle': 90.0, 'center': (0,0)}
    else:
        projectedSphere = {}
        projectedSphere['center'] = (result.projectedSphere.center[0] + frame.width / 2.0 ,frame.height / 2.0  -  result.projectedSphere.center[1])
        projectedSphere['axes'] =  (result.projectedSphere.minor_radius * 2.0 ,result.projectedSphere.major_radius * 2.0)
        #TODO result.projectedSphere.angle is always 0
        projectedSphere['angle'] = - (result.projectedSphere.angle * 180.0 / PI - 90.0)
    py_result['projected_sphere'] = projectedSphere

    py_result['model_confidence'] = result.modelConfidence
    py_result['model_id'] = result.modelID
    py_result['model_birth_timestamp'] = result.modelBirthTimestamp


    coords = cart2sph(result.circle.normal)
    if str(coords[0]) == 'nan':
        py_result['theta'] = 0
        py_result['phi'] = 0
    else:
        py_result['theta'] = coords[0]
        py_result['phi'] = coords[1]
    py_result['method'] = '3d c++'

    return py_result

cdef inline prepareForVisualization3D(  Detector3DResult& result ):

    py_visualizationResult = {}

    py_visualizationResult['circle'] = getCircle(result);
    py_visualizationResult['cost'] = result.cost
    py_visualizationResult['predicted_circle'] = getPredictedCircle(result);
    py_visualizationResult['refraction_result'] = convertTo3DPythonRefractionResult(result.RefractionResult)
    #py_visualizationResult['edges'] = getEdges(result)

    models = []
    for model in result.models:
        props = {}
        props['optimized_parameters'] = model.optimizedParameters
        props['cost_per_pupil'] = model.costPerPupil
        props['resFit'] = model.resFit
        props['sphere'] = getSphere(model)
        props['initial_sphere'] = getInitialSphere(model)
        props['model_id'] = model.modelID
        props['birth_timestamp'] = model.birthTimestamp
        models.append(props)
        #props['bin_positions'] = getBinPositions(model)
        #props['maturity'] = model.maturity
        #props['solver_fit'] = model.solverFit
        #props['confidence'] = model.confidence
        #props['performance'] = model.performance
        #props['performance_gradient'] = model.performanceGradient

    py_visualizationResult['models'] = models;

    return py_visualizationResult

cdef inline getBinPositions( ModelDebugProperties& result ):
    if result.binPositions.size() == 0:
        return []
    positions = []
    eyePosition = result.sphere.center
    eyeRadius = result.sphere.radius
    #bins are on a unit sphere
    for point in result.binPositions:
        positions.append([point[0]*eyeRadius+eyePosition[0],point[1]*eyeRadius+eyePosition[1],point[2]*eyeRadius+eyePosition[2]])
    return positions

cdef inline getFinalEdges2D( Detector2DResult& result ):
    if result.final_edges.size() == 0:
        return []
    edges = []
    for point in result.final_edges:
        edges.append([point.x,point.y])
    return edges

cdef inline getRawEdges2D( Detector2DResult& result ):
    if result.raw_edges.size() == 0:
        return []
    edges = []
    for point in result.raw_edges:
        edges.append([point.x,point.y])
    return edges


cdef inline getEdges( Detector3DResult& result):
    if result.edges.size() == 0:
        return []
    edges = []
    for point in result.edges:
        edges.append([point[0],point[1],point[2]])
    return edges

cdef inline getCircle(const Detector3DResult& result):
    center = result.circle.center
    radius = result.circle.radius
    normal = result.circle.normal
    return [ [center[0],center[1],center[2]], [normal[0],normal[1],normal[2]], radius ]

cdef inline getPredictedCircle(const Detector3DResult& result):
    center = result.predictedCircle.center
    radius = result.predictedCircle.radius
    normal = result.predictedCircle.normal
    return [ [center[0],center[1],center[2]], [normal[0],normal[1],normal[2]], radius ]

cdef inline getSphere(const ModelDebugProperties& result ):
    sphere = result.sphere
    return [ [sphere.center[0],sphere.center[1],sphere.center[2]],sphere.radius]

cdef inline getInitialSphere(const ModelDebugProperties& result ):
    sphere = result.initialSphere
    return [ [sphere.center[0],sphere.center[1],sphere.center[2]],sphere.radius]
