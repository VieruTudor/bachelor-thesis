using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace EmotionDetection.DetectionSystem.Utils
{
    public static class Deserializer
    {
        public static async Task<T> GetHttpResponseResult<T>(HttpResponseMessage message)
        {
            var responseBody = await message.Content.ReadAsStringAsync();
            var response = JsonConvert.DeserializeObject<T>(responseBody);

            return response;
        }
    }
}
