using UnityEngine;
using System.Collections;

public class AgentUnit : MonoBehaviour
{
    private bool isMoving = false;
    private string pendingAction = null;
    private bool hasAction = false;
    private Rigidbody rb;
    
    // Physics-based movement variables (inspired by AgentController)
    private float currentSpeed = 0f;
    private float targetSpeed = 0f;
    private Vector3 currentDirection = Vector3.zero;
    private Vector3 targetDirection = Vector3.zero;
    private float acceleration = 3f;
    private float directionChangeSpeed = 5f;
    
    [SerializeField] private float moveForce = 40f;
    [SerializeField] private float moveDuration = 0f;
    
    [Header("Death Detection")]
    public GameObject ground;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        
        // Physics settings inspired by AgentController
        rb.linearDamping = 0f; // No linear drag
        rb.angularDamping = 0f; // No angular drag
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        
        FindObjectOfType<AgentManager>().RegisterAgent(this);
    }
    


    void Update()
    {
        // Apply continuous movement based on current action (inspired by AgentController)
        ApplyContinuousMovement();
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
        
        // Check for death and reset immediately after API call
        if (ground != null)
        {
            float distance = Vector3.Distance(transform.position, ground.transform.position);
            if (distance < 1f) // Adjust threshold as needed
            {
                StadiumManager stadiumManager = FindObjectOfType<StadiumManager>();
                if (stadiumManager != null)
                {
                    stadiumManager.OnAgentDeath(gameObject.name);
                }
            }
        }
    }

    public void ExecuteMove()
    {
        if (!hasAction) return;

        isMoving = true;
        Vector3 direction = DirectionToVector(pendingAction);
        
        // Use continuous movement system for bigger strides
        SetMovementTarget(direction, 8f); // Reduced speed for more controlled movement
        
        // Set a timer to stop the movement
        StartCoroutine(StopMovementAfterDelay());
        
        hasAction = false;
        pendingAction = null;
    }

    private IEnumerator StopMovementAfterDelay()
    {
        // Wait for the movement duration
        yield return new WaitForSeconds(0.2f); // Adjust this for stride length
        
        // Stop the movement
        SetMovementTarget(Vector3.zero, 0f);
        isMoving = false;
    }

    private IEnumerator MoveWithImpulse(Vector3 direction)
    {
        // Apply a single impulse to start the movement
        rb.AddForce(direction * moveForce, ForceMode.Impulse);
        
        // Wait for the movement duration
        yield return new WaitForSeconds(moveDuration);
        
        isMoving = false;
    }

    // Continuous movement system inspired by AgentController
    private void ApplyContinuousMovement()
    {
        // Gradually adjust current speed towards target speed (like car acceleration/braking)
        if (currentSpeed < targetSpeed)
        {
            currentSpeed = Mathf.Min(currentSpeed + acceleration * Time.deltaTime, targetSpeed);
        }
        else if (currentSpeed > targetSpeed)
        {
            currentSpeed = Mathf.Max(currentSpeed - acceleration * Time.deltaTime, targetSpeed);
        }
        
        // Gradually adjust current direction towards target direction
        if (Vector3.Distance(currentDirection, targetDirection) > 0.1f)
        {
            currentDirection = Vector3.Lerp(currentDirection, targetDirection, directionChangeSpeed * Time.deltaTime);
            currentDirection.Normalize();
        }
        else
        {
            currentDirection = targetDirection;
        }
        
        // Apply the current speed in the current direction while preserving Y velocity for gravity
        Vector3 targetVelocity = currentDirection * currentSpeed;
        rb.linearVelocity = new Vector3(targetVelocity.x, rb.linearVelocity.y, targetVelocity.z);
    }

    // Enhanced movement method that sets target direction and speed
    private void SetMovementTarget(Vector3 direction, float speed)
    {
        if (direction.magnitude > 0.1f)
        {
            targetDirection = direction.normalized;
            targetSpeed = speed;
        }
        else
        {
            targetSpeed = 0f; // Stop if no movement
        }
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
