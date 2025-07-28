using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Text;

public static class MLAPI
{
    private static string baseUrl = "http://127.0.0.1:5001";

    public static IEnumerator RequestNextAction(float[] state, float reward, Action<string> callback)
    {
        string jsonPayload = JsonUtility.ToJson(new StateRequest { state = state, reward = reward });

        UnityWebRequest request = UnityWebRequest.Post($"{baseUrl}/get_next_action", jsonPayload, "application/json");
        request.SetRequestHeader("Accept", "application/json");
        request.timeout = 60; // tolerate up to 60 seconds

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"[MLAPI] Request failed: {request.error}. Terminating game.");
            // callback?.Invoke(null);
            // #if UNITY_EDITOR
            //     UnityEditor.EditorApplication.isPlaying = false;
            // #else
            //     Application.Quit();
            // #endif
            // yield break;
        }

        DirectionResponse response = JsonUtility.FromJson<DirectionResponse>(request.downloadHandler.text);
        callback?.Invoke(response.action.Trim().ToLower());
    }
}
