using UnityEngine;
using System.Collections;

public class AgentUnit : MonoBehaviour
{
    private bool isMoving = false;
    private string pendingAction = null;
    private bool hasAction = false;
    
    void Start()
    {
        FindObjectOfType<AgentManager>().RegisterAgent(this);
    }

    public bool IsBusy() => isMoving;

    public IEnumerator RequestAction()
    {
        Vector3 moveDir = Vector3.zero;
        bool gotResponse = false;

        // Get actual state and reward from components
        float[] state = GetComponent<StateFunction>().GetState();
        float reward = GetComponent<RewardFunction>().CalculateReward();

        yield return MLAPI.RequestNextAction(state, reward, direction =>
        {
            pendingAction = direction;
            gotResponse = true;
        });

        yield return new WaitUntil(() => gotResponse);
        hasAction = true;
    }

    public void ExecuteMove()
    {
        if (!hasAction) return;

        isMoving = true;
        StartCoroutine(MoveSmooth(DirectionToVector(pendingAction)));
        hasAction = false;
        pendingAction = null;
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
        isMoving = false;
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
