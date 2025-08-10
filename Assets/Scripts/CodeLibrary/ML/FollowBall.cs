using UnityEngine;

public class FollowBall : MonoBehaviour
{
    public Transform ball;                // assign the rolling sphere
    public bool preserveInitialYaw = true;

    private Vector3 positionOffset;       // cube - ball offset
    private float initialYawOffset;       // yaw difference at start

    void Start()
    {
        if (!ball) return;

        // Cache initial offsets so it stays where you placed it
        positionOffset = transform.position - ball.position;

        float ballYaw = ball.eulerAngles.y;
        float cubeYaw = transform.eulerAngles.y;
        initialYawOffset = Mathf.DeltaAngle(ballYaw, cubeYaw);
    }

    void LateUpdate()
    {
        if (!ball) return;

        // Follow position
        transform.position = ball.position + positionOffset;

        // Temporarily disabled rotation - keep original rotation
        // Copy only Y rotation from the ball
        // float ballYaw = ball.eulerAngles.y;
        // float targetYaw = preserveInitialYaw ? ballYaw + initialYawOffset : ballYaw;
        // transform.rotation = Quaternion.Euler(0f, targetYaw, 0f);
    }
}
