using UnityEngine;
using System.Collections;

public class AgentUnit : MonoBehaviour
{
    private bool isMoving = false;

    void Start()
    {
        FindObjectOfType<AgentManager>().RegisterAgent(this);
    }

    public bool IsBusy() => isMoving;

    public IEnumerator RequestAndMove()
    {
        isMoving = true;

        Vector3 moveDir = Vector3.zero;
        bool gotResponse = false;

        // Pause game
        Time.timeScale = 0f;

        yield return MLAPI.RequestNextAction(new float[] { 0f }, 0f, direction =>
        {
            moveDir = DirectionToVector(direction);
            gotResponse = true;
        });

        yield return new WaitUntil(() => gotResponse);

        // Resume game
        Time.timeScale = 1f;

        // Move if valid
        if (moveDir != Vector3.zero)
            yield return MoveSmooth(moveDir);

        isMoving = false;
    }


    private IEnumerator MoveSmooth(Vector3 direction)
    {
        Vector3 start = transform.position;
        Vector3 end = start + direction;
        float duration = 0.25f;
        float t = 0;

        while (t < 1f)
        {
            t += Time.deltaTime / duration;
            transform.position = Vector3.Lerp(start, end, t);
            yield return null;
        }

        transform.position = end;
    }

    private Vector3 DirectionToVector(string dir)
    {
        return dir switch
        {
            "up" => Vector3.forward,
            "down" => Vector3.back,
            "left" => Vector3.left,
            "right" => Vector3.right,
            _ => Vector3.zero
        };
    }
}
