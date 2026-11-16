import Foundation
import Testing
@testable import PiSwiftAI

private actor OpenAICacheCaptureClient: ProviderHTTPClient {
    var body: Data?
    func send(_ request: URLRequest) async throws -> ProviderHTTPResponse {
        body = request.httpBody
        return ProviderHTTPResponse(statusCode: 400, body: Data())
    }
}

@Suite("OpenAI cache serialization")
struct OpenAICacheSerializationTests {
    @Test(arguments: [Api.openAICompletions, .openAIResponses, .azureOpenAIResponses, .openAICodexResponses])
    func replayedToolArgumentsAreStable(api: Api) async throws {
        let model = Model(id: "test", name: "Test", api: api, provider: "openai",
            baseUrl: "https://cache-test.invalid/v1", reasoning: false, input: [.text],
            cost: .init(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
            contextWindow: 8192, maxTokens: 1024)
        var arguments = Set<String>()
        for attempt in 0..<16 {
            let fields = [("oldText", "before"), ("newText", "after"), ("context", "setup")]
            let edit = Dictionary(uniqueKeysWithValues: attempt.isMultiple(of: 2) ? fields : fields.reversed())
            let call = ToolCall(id: "call_1", name: "edit", arguments: [
                "path": AnyCodable("Main.lua"), "edits": AnyCodable([edit])
            ])
            let assistant = AssistantMessage(content: [.toolCall(call)], api: api,
                provider: model.provider, model: model.id,
                usage: .init(input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0), stopReason: .toolUse)
            let context = Context(messages: [.assistant(assistant)])
            if api == .openAICompletions {
                let client = OpenAICacheCaptureClient()
                _ = await streamOpenAICompletions(model: model, context: normalizeContext(context),
                    options: OpenAICompletionsOptions(apiKey: "test", httpClient: client, maxRetries: 0)).result()
                let body = try #require(await client.body)
                let payload = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
                let messages = try #require(payload["messages"] as? [[String: Any]])
                let message = try #require(messages.first { $0["role"] as? String == "assistant" })
                let calls = try #require(message["tool_calls"] as? [[String: Any]])
                let function = try #require(calls.first?["function"] as? [String: Any])
                arguments.insert(try #require(function["arguments"] as? String))
            } else {
                let input: [[String: Any]]
                if api == .openAICodexResponses {
                    input = try #require(convertCodexMessages(model: model, context: context) as? [[String: Any]])
                } else {
                    let query = try api == .azureOpenAIResponses
                        ? buildAzureResponsesQuery(model: model, context: context,
                            options: AzureOpenAIResponsesOptions(), deploymentName: model.id)
                        : buildResponsesQuery(model: model, context: context, options: OpenAIResponsesOptions())
                    let payload = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(query)) as? [String: Any])
                    input = try #require(payload["input"] as? [[String: Any]])
                }
                let function = try #require(input.first { $0["type"] as? String == "function_call" })
                arguments.insert(try #require(function["arguments"] as? String))
            }
        }
        // These strings are model-visible history, even when the outer request JSON is reordered.
        #expect(arguments.count == 1)
    }
}
