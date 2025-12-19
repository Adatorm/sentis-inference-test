using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class Executor : MonoBehaviour
{
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private TMP_Text inferenceTimeText;
    private Model _model;
    private bool _isInitialized = false;
    private bool _isRunning = false;
    private Worker _worker;
    private Tensor<float> _inputTensor;
    private int _executionCount = 0;
    private const int MaxExecution = 5;
    private IEnumerator _schedule;
    private const int LayersPerFrame = 5;
    private bool _isCompleted = false;
    private Stopwatch _stopwatch = new Stopwatch();
    private List<List<double>> _inferenceTimes = new List<List<double>>();
    private List<double> _currentList = new List<double>();

    private void Start()
    {
        _model = ModelLoader.Load(modelAsset);
        _worker = new Worker(_model, BackendType.GPUCompute);

        var inputShape = _model.inputs[0].shape.ToTensorShape();
        
        _inputTensor = new Tensor<float>(inputShape);
        
        _isInitialized = true;
    }

    private void Update()
    {
        if (!_isInitialized || _isCompleted) return;
        if (!_isRunning)
        {
            if (_executionCount >= MaxExecution)
            {
                // stop at max execution
                _isCompleted = true;
                DisplayTimes();
                return;
            }
            
            _schedule = _worker.ScheduleIterable(_inputTensor);
            _isRunning = true;
            _executionCount++;
            _currentList = new List<double>();
            _inferenceTimes.Add(_currentList);
        }
        
        _stopwatch.Restart();
        int it = 0;
        while (_schedule.MoveNext())
        {
            if (++it % LayersPerFrame == 0)
            {
                _stopwatch.Stop();
                _currentList.Add(_stopwatch.Elapsed.TotalMilliseconds);
                return;
            }
        }
        _stopwatch.Stop();
        _currentList.Add(_stopwatch.Elapsed.TotalMilliseconds);
        
        var outputTensor = _worker.PeekOutput() as Tensor<float>;

        // If you wish to read from the tensor, download it to cpu.
        var cpuTensor = outputTensor.ReadbackAndClone();
        
        var output = cpuTensor.DownloadToArray();
        cpuTensor.Dispose();
        _isRunning = false;
    }
    

    private void DisplayTimes()
    {
        var ss = new StringBuilder();
        for (int i = 0; i < _inferenceTimes.Count; i++)
        {
            ss.AppendFormat($"{i:00} - layer times: ");
            double total = 0.0;
            for (int j = 0; j < _inferenceTimes[i].Count; j++)
            {
                total += _inferenceTimes[i][j];
                //ss.AppendFormat($"{_inferenceTimes[i][j]:F2} ms, ");
            }
            Debug.Log($"{i:00} - total time: {total:F2} ms");
            ss.AppendFormat($"total:{total:F2} ms");
            ss.AppendLine();

        }
        inferenceTimeText.text = ss.ToString();
    }
    
    
    private void OnDestroy()
    {
        if (_isInitialized)
        {
            _worker.Dispose();
            _inputTensor.Dispose();
            _isInitialized = false;
        }
    }
}
