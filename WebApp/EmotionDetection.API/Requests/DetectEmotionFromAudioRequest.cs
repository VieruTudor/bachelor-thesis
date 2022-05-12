using EmotionDetection.API.Utils;
using FluentValidation;
using Microsoft.AspNetCore.Http;

namespace EmotionDetection.API.Requests
{
    public class DetectEmotionFromAudioRequest
    {
        public IFormFile AudioFile { get; set; }
    }

    public class DetectEmotionFromAudioRequestValidator : AbstractValidator<DetectEmotionFromAudioRequest>
    {
        public DetectEmotionFromAudioRequestValidator()
        {
            RuleFor(x => x.AudioFile).NotNull();
            RuleFor(x => x.AudioFile).SetValidator(new FileValidator());
        }
    }

    public class FileValidator : AbstractValidator<IFormFile>
    {
        public FileValidator()
        {
            ClassLevelCascadeMode = CascadeMode.Stop;

            RuleFor(x => x).Must(FileUtils.HasFileName).WithMessage("File name must not be empty.");
            RuleFor(x => x).Must(x => !FileUtils.HasUnsupportedCharacters(x)).WithMessage("File name must not contain unsupported characters.");
            RuleFor(x => x).Must(FileUtils.HasContent).WithMessage("File content must not be empty.");
            RuleFor(x => x).Must(FileUtils.BeSupportedFile).WithMessage("The file's extension is not supported.");
            RuleFor(x => x).Must(FileUtils.ContentMatchExtension).WithMessage("The file's extension does not match the content type.");
        }
    }

}
