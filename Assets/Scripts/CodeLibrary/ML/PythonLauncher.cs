using System.Diagnostics;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Networking;
using System.Collections;
using System.IO;


public class PythonLauncher : MonoBehaviour
{
    private Process apiProcess;

    void Start()
    {
        Time.timeScale = 0f; // Pause game
        StartCoroutine(LaunchSequence());
    }

    IEnumerator LaunchSequence()
    {
        yield return RunPortCleanup();
        LaunchAPI();
        yield return WaitForAPIReady("http://127.0.0.1:5000/ping");

        Time.timeScale = 1f; // Unpause game
        SceneManager.LoadScene("TestScene"); // Replace with your actual game scene name
    }

    IEnumerator RunPortCleanup()
    {
        string bashPath = "/bin/bash";
        string scriptPath = "/Users/rishigaddamanugu/Desktop/Unity Projects/Sumo AI/Assets/Scripts/CodeLibrary/ML/clean_ports.sh";

        ProcessStartInfo cleanInfo = new ProcessStartInfo
        {
            FileName = bashPath,
            Arguments = $"\"{scriptPath}\"",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        using (Process cleanProcess = new Process())
        {
            cleanProcess.StartInfo = cleanInfo;
            cleanProcess.Start();
            cleanProcess.WaitForExit();
            UnityEngine.Debug.Log("Port cleanup finished");
        }

        yield return null;
    }

    void LaunchAPI()
    {
        string pythonPath = "/opt/homebrew/Caskroom/miniconda/base/envs/unity_env/bin/python3";
        string scriptPath = "/Users/rishigaddamanugu/Desktop/Unity Projects/Sumo AI/Assets/Scripts/CodeLibrary/ML/app.py";
        string logFilePath = "/Users/rishigaddamanugu/Desktop/Unity Projects/Sumo AI/flask_debug.log"; // Or use Application.persistentDataPath

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"\"{scriptPath}\"",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        apiProcess = new Process
        {
            StartInfo = startInfo
        };

        // Redirect output to file
        StreamWriter logWriter = new StreamWriter(logFilePath, append: false); // overwrite on each run
        apiProcess.OutputDataReceived += (sender, args) =>
        {
            if (!string.IsNullOrEmpty(args.Data))
            {
                logWriter.WriteLine($"[STDOUT] {args.Data}");
                logWriter.Flush();
            }
        };

        apiProcess.ErrorDataReceived += (sender, args) =>
        {
            if (!string.IsNullOrEmpty(args.Data))
            {
                logWriter.WriteLine($"[STDERR] {args.Data}");
                logWriter.Flush();
            }
        };

        apiProcess.Start();
        apiProcess.BeginOutputReadLine();
        apiProcess.BeginErrorReadLine();

        UnityEngine.Debug.Log($"API launched! Output logging to: {logFilePath}");
    }


    IEnumerator WaitForAPIReady(string url, float timeoutSeconds = 10f)
    {
        float startTime = Time.realtimeSinceStartup;

        while (Time.realtimeSinceStartup - startTime < timeoutSeconds)
        {
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                yield return request.SendWebRequest();

#if UNITY_2020_1_OR_NEWER
                if (request.result == UnityWebRequest.Result.Success && request.downloadHandler.text == "pong")
#else
                if (!request.isNetworkError && !request.isHttpError && request.downloadHandler.text == "pong")
#endif
                {
                    UnityEngine.Debug.Log("API responded with pong!");
                    yield break;
                }
            }

            yield return new WaitForSecondsRealtime(0.5f);
        }

        UnityEngine.Debug.LogError("API failed to respond in time.");
    }

    void OnApplicationQuit()
    {
        if (apiProcess != null && !apiProcess.HasExited)
        {
            apiProcess.Kill();
            apiProcess.Dispose();
        }
    }
}
