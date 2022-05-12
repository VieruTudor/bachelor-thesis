using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using EmotionDetection.API.Requests;
using EmotionDetection.API.Utils;
using EmotionDetection.Data.Entities;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace EmotionDetection.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class FileController : ControllerBase
    {
        private readonly IMediator _mediator;
        public FileController(IMediator mediator)
        {
            _mediator = mediator;
        }

        [HttpPost]
        [Route("detect-emotion-from-audio")]
        public async Task<ActionResult<List<Prediction>>> DetectEmotionFromAudio([FromForm] DetectEmotionFromAudioRequest request)
        {
            var command = await request.ToCommand();
            var result = await _mediator.Send(command);

            return Ok(result);
        }
    }
}
