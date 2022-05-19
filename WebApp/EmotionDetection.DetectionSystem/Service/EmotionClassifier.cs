using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using EmotionDetection.Data.Entities;
using EmotionDetection.Data.Enums;
using EmotionDetection.DetectionSystem.Configuration;
using EmotionDetection.DetectionSystem.Requests;
using EmotionDetection.DetectionSystem.Utils;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;

namespace EmotionDetection.DetectionSystem.Service
{
    public class EmotionClassifier : IEmotionClassifier
    {
        private HttpClient _httpClient;
        private string _serviceBaseUri;

        public EmotionClassifier(IOptions<EmotionClassifierSettings> settings)
        {
            _httpClient = new HttpClient();
            _serviceBaseUri = settings.Value.BaseApiUri;
        }

        public async Task<List<Prediction>> GetPredictionFromAudio(string audioPath)
        {
            var requestBody = new DetectEmotionFromInferenceModelRequest { AudioPath = audioPath };
            var detectAudioUri = $"{_serviceBaseUri}/detect-emotion-from-audio";

            var fullRequestBody = JsonConvert.SerializeObject(requestBody);
            var response = await _httpClient.PostAsJsonAsync(detectAudioUri, fullRequestBody);

            var results = Deserializer.GetHttpResponseResult<Dictionary<string, Dictionary<int, float>>>(response);

            return results.Result["results"]
                .Select(kvp =>
                    new Prediction
                    {
                        Emotion = (Emotion)kvp.Key,
                        Level = kvp.Value
                    })
                .ToList();
        }

        public List<Prediction> GetPredictionFromVideo(string videoPath)
        {
            throw new NotImplementedException();
        }
    }

    public sealed class CommandLineProcess : IDisposable
    {
        public string Path { get; }
        public string Arguments { get; }
        public bool IsRunning { get; private set; }
        public int? ExitCode { get; private set; }

        private Process Process;
        private readonly object Locker = new object();

        public CommandLineProcess(string path, string arguments)
        {
            Path = path ?? throw new ArgumentNullException(nameof(path));
            if (!File.Exists(path)) throw new ArgumentException($"Executable not found: {path}");
            Arguments = arguments;
        }

        public int Run(out string output, out string err)
        {
            lock (Locker)
            {
                if (IsRunning) throw new Exception("The process is already running");

                Process = new Process()
                {
                    EnableRaisingEvents = true,
                    StartInfo = new ProcessStartInfo()
                    {
                        FileName = Path,
                        Arguments = Arguments,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                    },
                };

                if (!Process.Start()) throw new Exception("Process could not be started");
                output = Process.StandardOutput.ReadToEnd();
                err = Process.StandardError.ReadToEnd();
                Process.WaitForExit();
                try
                {
                    Process.Refresh();
                }
                catch
                {
                }

                return (ExitCode = Process.ExitCode).Value;
            }
        }

        public void Kill()
        {
            lock (Locker)
            {
                try
                {
                    Process?.Kill();
                }
                catch
                {
                }

                IsRunning = false;
                Process = null;
            }
        }

        public void Dispose()
        {
            try
            {
                Process?.Dispose();
            }
            catch
            {
            }
        }
    }
}