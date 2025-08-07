using UnityEngine;
using System.Collections;

public class AgentUnit : MonoBehaviour
{
    private bool isDead = false;
    private bool isMoving = false;
    private string pendingAction = null;
    private bool hasAction = false;
    private Rigidbody rb;
    
    
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float turnSpeed = 100f;
    [SerializeField] private float moveDuration = 0f;
    
    [Header("Death Detection")]
    public GameObject ground;
    
    [Header("Ground Detection")]
    [SerializeField] private float groundCheckDistance = 0.1f;
    

    public GameObject platform;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        rb.linearDamping = 0.01f;
        rb.angularDamping = 1f;
        FindObjectOfType<AgentManager>().RegisterAgent(this);
    }
    
    void Update()
    {
        CheckOnPlatform();
    }
    
    private bool CheckOnPlatform()
    {
        if (platform == null) return false;
        
        // Get the distance between the agent and the platform
        float distanceToPlatform = Mathf.Abs(transform.position.y - platform.transform.position.y);
        
        // Simple threshold check - if agent is within this distance, consider it "on" the platform
        bool isOnPlatform = distanceToPlatform <= 1.2f; // Adjust this value as needed
        
        Debug.Log("Distance to platform: " + distanceToPlatform);
        return isOnPlatform;
    }
    

    public bool IsDead() => isDead;

    public void SetDeathState(bool state)
    {
        isDead = state;
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
        StartCoroutine(PerformAction(pendingAction));
        
        hasAction = false;
        pendingAction = null;

        // Check for death and reset immediately after API call
        if (ground != null)
        {
            float groundY = ground.transform.position.y;
            float agentY = transform.position.y;
            float heightDistance = Mathf.Abs(agentY - groundY);
            // Had to adjust threshold since ML model was double counting death on the fall to ground
            if (heightDistance < 1.19f) // Adjust threshold as needed
            {
                isDead = true;
            }
        }
    }

    private IEnumerator PerformAction(string action)
    {
        if (!CheckOnPlatform())
        {
            isMoving = false;
            yield break;
        }

        float elapsed = 0f;

        Vector3 moveDirection = Vector3.zero;
        float totalTurnAngle = 0f;

        switch (action)
        {
            case "forward":
                rb.AddForce(transform.forward * moveSpeed, ForceMode.VelocityChange);
                break;

            case "backward":
                rb.AddForce(-transform.forward * moveSpeed, ForceMode.VelocityChange);
                break;

            case "turnleft":
                rb.AddTorque(Vector3.up * -turnSpeed * 0.1f, ForceMode.VelocityChange);
                break;

            case "turnright":
                rb.AddTorque(Vector3.up * turnSpeed * 0.1f, ForceMode.VelocityChange);
                break;
        }

        // All actions are now instant physics-based, no need to wait
        yield return null;

        isMoving = false;
    }

}
