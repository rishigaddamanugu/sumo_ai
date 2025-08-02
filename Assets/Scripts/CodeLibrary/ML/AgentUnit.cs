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
        CheckGrounded();
    }
    
    private bool CheckGrounded()
    {
        // Cast a ray downward from the bottom of the cube
        Vector3 raycastOrigin = transform.position - new Vector3(0, GetComponent<Collider>().bounds.extents.y, 0);
        RaycastHit hit;
        bool isGrounded = Physics.Raycast(raycastOrigin, Vector3.down, out hit, groundCheckDistance);
        
        return isGrounded;
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
        if (!CheckGrounded())
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
                moveDirection = transform.forward;
                break;

            case "backward":
                moveDirection = -transform.forward;
                break;

            case "turnleft":
                totalTurnAngle = -turnSpeed * moveDuration;
                break;

            case "turnright":
                totalTurnAngle = turnSpeed * moveDuration;
                break;
        }

        while (elapsed < moveDuration)
        {
            float delta = Time.deltaTime;
            elapsed += delta;

            if (moveDirection != Vector3.zero)
                transform.position += moveDirection * moveSpeed * delta;

            if (Mathf.Abs(totalTurnAngle) > 0f)
            {
                float angleStep = (totalTurnAngle / moveDuration) * delta;
                transform.Rotate(Vector3.up, angleStep);
            }

            yield return null;
        }

        isMoving = false;
    }

}
