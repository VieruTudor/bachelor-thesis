using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using PommaLabs.MimeTypes;

namespace EmotionDetection.API.Utils
{
    public static class FileUtils
    {
        private static readonly Dictionary<string, List<string>> ExtensionsWithMime = new()
        {
            { "mp4", new() { "application/mp4" } },
            { "avi", new() { "video/x-msvideo" } },
            { "mpeg", new() { "video/mpeg", "audio/mpeg" } },
        };

        public static async Task<Stream> ToMemoryStream(IFormFile file)
        {
            var stream = new MemoryStream();

            if (file is null || file.ContentType == null || file.Length == 0)
            {
                return stream;
            }

            await file.CopyToAsync(stream);
            return stream;
        }

        public static List<Stream> ToMemoryStream(List<IFormFile> files)
        {
            if (files is null || files.Count == 0)
            {
                return new();
            }

            var streams = new List<Stream>();
            Stream newStream = new MemoryStream();

            files.ForEach(x =>
            {
                newStream = new MemoryStream();
                x.CopyTo(newStream);
                streams.Add(newStream);
            });

            return streams;
        }

        public static decimal BytesToGB(decimal bytes)
        {
            return bytes / 1073741824;
        }

        public static string GetExtension(IFormFile file)
        {
            return file.FileName.Split('.').LastOrDefault();
        }

        public static bool ContentMatchExtension(IFormFile file)
        {
            var extensionMimeTypes = MimeTypeMap.GetMimeTypes(file.FileName);
            var definedMimeTypes = ExtensionsWithMime[GetExtension(file)];

            MimeTypeMap.TryGetMimeType(file.OpenReadStream(), out var detectedMimeType);

            extensionMimeTypes = extensionMimeTypes
                .Union(definedMimeTypes)
                .ToList();

            return extensionMimeTypes.Contains(detectedMimeType);
        }

        public static bool BeSupportedFile(IFormFile file)
        {
            return ExtensionsWithMime.ContainsKey(GetExtension(file));
        }

        public static bool HasContent(IFormFile file)
        {
            return file.Length > 0;
        }

        public static bool HasFileName(IFormFile file)
        {
            var fileNameParts = file.FileName.Trim().Split('.').Where(x => x != "").ToList();
            var fileNameWithoutExtension = fileNameParts.Take(fileNameParts.Count - 1);

            return fileNameWithoutExtension.Count() != 0;
        }

        public static bool HasUnsupportedCharacters(IFormFile file)
        {
            return file.FileName.Any(Path.GetInvalidFileNameChars().Contains);
        }

        public static async Task<string> GetTempFilePath(IFormFile formFile)
        {
            var filePath = Path.GetTempFileName();

            await using var stream = File.Create(filePath);
            await formFile.CopyToAsync(stream);

            return filePath;
        }
    }
}
