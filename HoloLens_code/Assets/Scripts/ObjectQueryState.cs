﻿#if WINDOWS_UWP || DOTNETWINRT_PRESENT
#define SPATIALCOORDINATESYSTEM_API_PRESENT
#endif

using Microsoft.Azure.ObjectAnchors;
using Microsoft.Azure.ObjectAnchors.Unity;
using System;
using System.Threading.Tasks;
using UnityEngine;

public class ObjectQueryState : MonoBehaviour
{
    public ObjectQuery Query;
    public Material EnvironmentMaterial;
    private MeshFilter _meshFilter;
    private DateTime? _lastMeshUpdateTime = null;
    private bool _updateInProgress = false;
    public GameObject logger;
    private DebugOutput dbg;

    private void OnDestroy()
    {
        Query?.Dispose();
        Query = null;
    }

    // Start is called before the first frame update
    void Start()
    {
        _meshFilter = gameObject.AddComponent<MeshFilter>();
        _meshFilter.mesh = new Mesh();
        gameObject.AddComponent<MeshRenderer>().sharedMaterial = EnvironmentMaterial;
        if (dbg == null)
        {
            dbg = logger.GetComponent<DebugOutput>();
        }
        Debug.Log("From objectquerystate");
    }

    // Update is called once per frame
    async void Update()
    {
        if (!_updateInProgress &&
            Query != null &&
            EnvironmentMaterial != null &&
            (!_lastMeshUpdateTime.HasValue || DateTime.Now - _lastMeshUpdateTime.Value > TimeSpan.FromSeconds(2)))
        {
            _updateInProgress = true;
            EnvironmentObservation observation = await Query.ComputeLatestEnvironmentObservationAsync(EnvironmentObservationTopology.PointCloud);

            MeshLoader.MeshData? meshData = null;
            ObjectAnchorsLocation? observationLocation = null;

            await Task.Run(() =>
            {
                var vertexPositions = new System.Numerics.Vector3[observation.VertexCount];
                var vertexNormals = new System.Numerics.Vector3[vertexPositions.Length];
                var triangleIndices = new uint[observation.TriangleIndexCount];
                observation.GetVertexPositions(vertexPositions);
                observation.GetVertexNormals(vertexNormals);
                observation.GetTriangleIndices(triangleIndices);
                meshData = new MeshLoader.MeshData(vertexPositions, vertexNormals, triangleIndices);

#if SPATIALCOORDINATESYSTEM_API_PRESENT
                observationLocation = observation.Origin.ToSpatialCoordinateSystem().TryGetTransformTo(ObjectAnchorsWorldManager.WorldOrigin)?.ToUnityLocation();
#endif
            });


            if (observationLocation.HasValue && meshData.HasValue)
            {
                MeshLoader.LoadMesh(_meshFilter.mesh, meshData.Value);
                transform.SetPositionAndRotation(observationLocation.Value.Position, observationLocation.Value.Orientation);
                dbg.Log($"{observationLocation.Value.Position}, {observationLocation.Value.Orientation}");
            }
            else
            {
                dbg.Log("ObservatoinLocation or meshdata does not have value");
            }

            _lastMeshUpdateTime = DateTime.Now;
            _updateInProgress = false;
        }
    }
}
