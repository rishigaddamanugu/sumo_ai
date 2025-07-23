using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Text;

public static class MLAPI
{
    private static string baseUrl = "http://127.0.0.1:5000";

    public static IEnumerator RequestNextAction(float[] state, float reward, Action<string> callback)
    {
        string jsonPayload = JsonUtility.ToJson(new StateRequest { state = state, reward = reward });

        UnityWebRequest request = UnityWebRequest.Post($"{baseUrl}/get_next_action", jsonPayload, "application/json");
        request.SetRequestHeader("Accept", "application/json");
        request.timeout = 10; // tolerate up to 10 seconds

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogWarning($"[MLAPI] Request failed: {request.error}");
            callback?.Invoke(null);
            yield break;
        }

        DirectionResponse response = JsonUtility.FromJson<DirectionResponse>(request.downloadHandler.text);
        callback?.Invoke(response.action.Trim().ToLower());
    }
}
