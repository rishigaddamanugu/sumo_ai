using UnityEngine;
using System.Collections;

public class AgentController : MonoBehaviour
{
    private float moveDuration = 0.25f;
    private float delayBetweenMoves = 0.3f;
    private bool isMoving = false;

    void Start()
    {
        StartCoroutine(QueryAndMoveLoop());
    }

    IEnumerator QueryAndMoveLoop()
    {
        while (true)
        {
            if (!isMoving)
            {
                float[] dummyState = new float[] { 0f, 0f, 0f }; // Replace with actual state
                float dummyReward = 0f;

                Time.timeScale = 0f;  // ⏸️ Pause the entire game

                yield return MLAPI.RequestNextAction(dummyState, dummyReward, direction =>
                {
                    Time.timeScale = 1f;  // ▶️ Resume game once response is received

                    if (direction == "pause")
                    {
                        Debug.Log("[Agent] Model is training — pausing movement.");
                        return;
                    }

                    Vector3 dirVector = DirectionToVector(direction);
                    if (dirVector != Vector3.zero)
                    {
                        StartCoroutine(MoveSmooth(dirVector, moveDuration));
                    }
                });

                yield return new WaitForSeconds(delayBetweenMoves);
            }

            yield return null;
        }
    }


    IEnumerator MoveSmooth(Vector3 direction, float duration)
    {
        isMoving = true;
        Vector3 start = transform.position;
        Vector3 end = start + direction;

        float elapsed = 0f;
        while (elapsed < duration)
        {
            transform.position = Vector3.Lerp(start, end, elapsed / duration);
            elapsed += Time.deltaTime;
            yield return null;
        }

        transform.position = end;
        isMoving = false;
    }

    Vector3 DirectionToVector(string direction)
    {
        return direction switch
        {
            "up" => Vector3.forward,
            "down" => Vector3.back,
            "left" => Vector3.left,
            "right" => Vector3.right,
            _ => Vector3.zero
        };
    }
}
