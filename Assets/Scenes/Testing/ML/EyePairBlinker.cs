using UnityEngine;
using System.Collections;

public class EyePairBlinker : MonoBehaviour
{
    [Header("Blink Settings")]
    public float blinkInterval = 3f;
    public float blinkDuration = 0.15f;
    public float closedScaleY = 0.1f;
    public float blinkRandomness = 0.5f;

    private Transform leftEye;
    private Transform rightEye;

    private Vector3 leftEyeOriginalScale;
    private Vector3 rightEyeOriginalScale;
    private bool isBlinking = false;

    void Start()
    {
        // Auto-find eye children
        leftEye = transform.Find("LeftEye");
        rightEye = transform.Find("RightEye");

        if (leftEye == null || rightEye == null)
        {
            Debug.LogWarning("[EyePairBlinker] LeftEye or RightEye child not found.");
            return;
        }

        leftEyeOriginalScale = leftEye.localScale;
        rightEyeOriginalScale = rightEye.localScale;

        StartCoroutine(BlinkRoutine());
    }

    IEnumerator BlinkRoutine()
    {
        while (true)
        {
            yield return new WaitForSeconds(blinkInterval + Random.Range(-blinkRandomness, blinkRandomness));

            if (!isBlinking)
            {
                StartCoroutine(Blink());
            }
        }
    }

    IEnumerator Blink()
    {
        isBlinking = true;

        Vector3 leftClosed = new Vector3(leftEyeOriginalScale.x, leftEyeOriginalScale.y * closedScaleY, leftEyeOriginalScale.z);
        Vector3 rightClosed = new Vector3(rightEyeOriginalScale.x, rightEyeOriginalScale.y * closedScaleY, rightEyeOriginalScale.z);

        // Close eyes
        float elapsed = 0f;
        while (elapsed < blinkDuration / 2f)
        {
            float t = elapsed / (blinkDuration / 2f);
            leftEye.localScale = Vector3.Lerp(leftEyeOriginalScale, leftClosed, t);
            rightEye.localScale = Vector3.Lerp(rightEyeOriginalScale, rightClosed, t);
            elapsed += Time.deltaTime;
            yield return null;
        }

        yield return new WaitForSeconds(0.05f); // Eyes stay closed briefly

        // Open eyes
        elapsed = 0f;
        while (elapsed < blinkDuration / 2f)
        {
            float t = elapsed / (blinkDuration / 2f);
            leftEye.localScale = Vector3.Lerp(leftClosed, leftEyeOriginalScale, t);
            rightEye.localScale = Vector3.Lerp(rightClosed, rightEyeOriginalScale, t);
            elapsed += Time.deltaTime;
            yield return null;
        }

        leftEye.localScale = leftEyeOriginalScale;
        rightEye.localScale = rightEyeOriginalScale;

        isBlinking = false;
    }
}
