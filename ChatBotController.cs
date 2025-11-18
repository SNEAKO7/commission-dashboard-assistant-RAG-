using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Commissions.Controllers
{
    [Route("api/chatbot")]
    [ApiController]
    public class ChatBotController : ControllerBase
    {
        private readonly IHttpClientFactory _httpClientFactory;
        
        public ChatBotController(IHttpClientFactory httpClientFactory)
        {
            _httpClientFactory = httpClientFactory;
        }

        public class MessageRequest
        {
            public string Message { get; set; }
            public string UserId { get; set; }      // ← ADD THIS
            public string SessionId { get; set; }   // ← ADD THIS
            public string ClientId { get; set; }    // ← ADD THIS (optional)
        }

        [HttpPost]
        public async Task<IActionResult> Post([FromBody] MessageRequest request)
        {
            var httpClient = _httpClientFactory.CreateClient();

            // ✅ FIXED: Send user_id and session_id to Flask
            var payload = JsonConvert.SerializeObject(new { 
                message = request.Message,
                user_id = request.UserId,        // ← PASS TO FLASK
                session_id = request.SessionId,  // ← PASS TO FLASK
                client_id = request.ClientId     // ← PASS TO FLASK
            });
            
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            // Create request to Flask
            var flaskRequest = new HttpRequestMessage(HttpMethod.Post, "http://localhost:5000/chat")
            {
                Content = content
            };

            // Get JWT from incoming request
            string jwt = null;
            if (Request.Headers.ContainsKey("Authorization"))
            {
                var rawHeader = Request.Headers["Authorization"].ToString();
                if (rawHeader.StartsWith("Bearer "))
                    jwt = rawHeader.Substring("Bearer ".Length);
            }

            // Forward JWT to Flask
            if (!string.IsNullOrEmpty(jwt))
                flaskRequest.Headers.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);

            // Send to Flask
            var flaskResponse = await httpClient.SendAsync(flaskRequest);
            var flaskJson = await flaskResponse.Content.ReadAsStringAsync();

            // Return Flask response
            return new ContentResult
            {
                Content = flaskJson,
                ContentType = "application/json",
                StatusCode = (int)flaskResponse.StatusCode
            };
        }

        // ✅ NEW: Add endpoint to get chat history
        [HttpGet("history")]
        public async Task<IActionResult> GetHistory([FromQuery] string userId, [FromQuery] string sessionId)
        {
            if (string.IsNullOrEmpty(userId) || string.IsNullOrEmpty(sessionId))
            {
                return BadRequest(new { error = "userId and sessionId are required" });
            }

            var httpClient = _httpClientFactory.CreateClient();

            // Get JWT
            string jwt = null;
            if (Request.Headers.ContainsKey("Authorization"))
            {
                var rawHeader = Request.Headers["Authorization"].ToString();
                if (rawHeader.StartsWith("Bearer "))
                    jwt = rawHeader.Substring("Bearer ".Length);
            }

            // Create request to Flask
            var flaskRequest = new HttpRequestMessage(
                HttpMethod.Get, 
                $"http://localhost:5000/chat/history?user_id={userId}&session_id={sessionId}"
            );

            // Forward JWT
            if (!string.IsNullOrEmpty(jwt))
                flaskRequest.Headers.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);

            // Send to Flask
            var flaskResponse = await httpClient.SendAsync(flaskRequest);
            var flaskJson = await flaskResponse.Content.ReadAsStringAsync();

            return new ContentResult
            {
                Content = flaskJson,
                ContentType = "application/json",
                StatusCode = (int)flaskResponse.StatusCode
            };
        }

        // ✅ NEW: Clear chat history
        [HttpPost("clear")]
        public async Task<IActionResult> ClearHistory([FromBody] MessageRequest request)
        {
            var httpClient = _httpClientFactory.CreateClient();

            var payload = JsonConvert.SerializeObject(new { 
                user_id = request.UserId,
                session_id = request.SessionId
            });
            
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var flaskRequest = new HttpRequestMessage(HttpMethod.Post, "http://localhost:5000/chat/clear")
            {
                Content = content
            };

            // Get and forward JWT
            string jwt = null;
            if (Request.Headers.ContainsKey("Authorization"))
            {
                var rawHeader = Request.Headers["Authorization"].ToString();
                if (rawHeader.StartsWith("Bearer "))
                    jwt = rawHeader.Substring("Bearer ".Length);
            }

            if (!string.IsNullOrEmpty(jwt))
                flaskRequest.Headers.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);

            var flaskResponse = await httpClient.SendAsync(flaskRequest);
            var flaskJson = await flaskResponse.Content.ReadAsStringAsync();

            return new ContentResult
            {
                Content = flaskJson,
                ContentType = "application/json",
                StatusCode = (int)flaskResponse.StatusCode
            };
        }
    }
}