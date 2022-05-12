using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using EmotionDetection.Data.Entities;
using EmotionDetection.Data.Enums;
using EmotionDetection.DetectionSystem.Configuration;
using Microsoft.Extensions.Options;

namespace EmotionDetection.DetectionSystem.Service
{
    public class EmotionClassifier : IEmotionClassifier
    {
        private string _modelPath;
        private readonly string _pythonPath;
        private ProcessStartInfo _processStartInfo;

        public EmotionClassifier(IOptions<EmotionClassifierSettings> settings)
        {
            var cwd = Directory.GetParent(Directory.GetCurrentDirectory()).Parent;

            _modelPath = cwd + settings.Value.ModelPath;
            _pythonPath = settings.Value.PythonPath;
            InitialiseProcessStartInfo();
        }

        private void InitialiseProcessStartInfo()
        {
            _processStartInfo = new ProcessStartInfo()
            {
                FileName = _pythonPath,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                // WorkingDirectory = System.IO.Directory.GetParent(_pythonPath).Name,
                CreateNoWindow = false
            };
        }

        public List<Prediction> GetPredictionFromAudio(string audioPath)
        {
            _processStartInfo.Arguments = $"{_pythonPath} -u {_modelPath} {audioPath}";

            // 0 0.1;1 0.01;2 0.0005

            var results = string.Empty;

            using (CommandLineProcess cmd = new CommandLineProcess(_pythonPath, $"-u {_modelPath} {audioPath}"))
            {
                var sb = new StringBuilder();
                // Call Python:
                var exitCode = cmd.Run(out var processOutput, out var processError);

                // Get result:
                sb.AppendLine(processOutput);
                sb.AppendLine(processError);
                results = sb.ToString();
            }

            var predictions = results.Substring(0, results.LastIndexOf(';')).Split(';')
                .Select(pred => new Prediction
                    { Emotion = (Emotion)int.Parse(pred.Split(' ')[0]), Level = float.Parse(pred.Split(' ')[1], CultureInfo.InvariantCulture.NumberFormat) })
                .ToList();
            return predictions;
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