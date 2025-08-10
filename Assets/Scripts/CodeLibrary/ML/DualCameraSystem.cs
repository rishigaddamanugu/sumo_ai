using UnityEngine;

public class DualCameraSystem : MonoBehaviour
{
    [Header("Main Camera Settings")]
    private Vector3 mainCameraPosition = new Vector3(0, 17, -16.5f);
    private Vector3 mainCameraRotation = new Vector3(10, 0, 0);
    private Vector3 mainCameraScale = Vector3.one;
    
    [Header("Orbiting Camera Settings")]
    public Transform centralPlatform; // Assign in inspector
    private float orbitRadius = 20f;
    private float orbitHeight = 15f;
    private float orbitSpeed = 20f; // Degrees per second
    
    [Header("Camera Switching")]
    private float mainCameraDuration = 10f; // Main camera stays for 10 seconds
    private float orbitingCameraDuration = 5f; // Orbiting camera stays for 5 seconds
    private bool enableAutoSwitch = true;
    
    private Camera mainCamera;
    private Camera orbitingCamera;
    private bool isOrbitingCameraActive = false;
    private float lastSwitchTime;
    private float orbitAngle = 0f;
    
    
    void Start()
    {
        SetupMainCamera();
        CreateOrbitingCamera();
        lastSwitchTime = Time.unscaledTime;
    }
    
    void LateUpdate()
    {
        // Check if it's time to switch cameras based on current camera duration
        float currentDuration = isOrbitingCameraActive ? orbitingCameraDuration : mainCameraDuration;
        if (enableAutoSwitch && Time.unscaledTime - lastSwitchTime >= currentDuration)
        {
            SwitchCamera();
            lastSwitchTime = Time.unscaledTime;
        }
        
        // Update orbiting camera position if it's active
        if (isOrbitingCameraActive)
        {
            UpdateOrbitingCameraPosition();
        }
    }
    
    private void SetupMainCamera()
    {
        // Get the main camera or create one if it doesn't exist
        mainCamera = Camera.main;
        if (mainCamera == null)
        {
            GameObject mainCameraGO = new GameObject("MainCamera");
            mainCamera = mainCameraGO.AddComponent<Camera>();
            mainCameraGO.tag = "MainCamera";
        }
        
        // Set the main camera properties
        mainCamera.transform.position = mainCameraPosition;
        mainCamera.transform.rotation = Quaternion.Euler(mainCameraRotation);
        mainCamera.transform.localScale = mainCameraScale;
        
        // Set camera properties for good viewing
        mainCamera.clearFlags = CameraClearFlags.Skybox;
        mainCamera.fieldOfView = 60f;
        
        // Initially enable the main camera
        mainCamera.enabled = true;
    }
    
    private void CreateOrbitingCamera()
    {
        // Create the orbiting camera GameObject
        GameObject orbitingCameraGO = new GameObject("OrbitingCamera");
        orbitingCamera = orbitingCameraGO.AddComponent<Camera>();
        
        // Set camera properties
        orbitingCamera.clearFlags = CameraClearFlags.Skybox;
        orbitingCamera.fieldOfView = 50f;
        
        // Initially disable the orbiting camera
        orbitingCamera.enabled = false;
        
        // Set initial position
        UpdateOrbitingCameraPosition();
    }
    
    private void UpdateOrbitingCameraPosition()
    {
        if (centralPlatform == null)
        {
            Debug.LogWarning("Central platform not assigned! Using origin as center.");
            return;
        }
        
        // Calculate orbit position around the central platform
        float x = centralPlatform.position.x + Mathf.Cos(orbitAngle * Mathf.Deg2Rad) * orbitRadius;
        float z = centralPlatform.position.z + Mathf.Sin(orbitAngle * Mathf.Deg2Rad) * orbitRadius;
        float y = centralPlatform.position.y + orbitHeight;
        
        orbitingCamera.transform.position = new Vector3(x, y, z);
        
        // Look at the central platform
        orbitingCamera.transform.LookAt(centralPlatform.position);
        
        // Update orbit angle (independent of game speed but stops when paused)
        if (Time.timeScale > 0f)
        {
            orbitAngle += orbitSpeed * Time.unscaledDeltaTime;
            if (orbitAngle >= 360f) orbitAngle -= 360f;
        }
    }
    
    private void SwitchCamera()
    {
        isOrbitingCameraActive = !isOrbitingCameraActive;
        
        if (isOrbitingCameraActive)
        {
            mainCamera.enabled = false;
            orbitingCamera.enabled = true;
            Debug.Log("🎬 Switched to ORBITING camera");
        }
        else
        {
            orbitingCamera.enabled = false;
            mainCamera.enabled = true;
            Debug.Log("🎬 Switched to MAIN camera");
        }
        

    }
    
    // Private methods for internal control
    private void SetOrbitSpeed(float speed)
    {
        orbitSpeed = speed;
    }
    
    private void SetOrbitRadius(float radius)
    {
        orbitRadius = radius;
    }
    
    private void SetOrbitHeight(float height)
    {
        orbitHeight = height;
    }
    
    private void ForceMainCamera()
    {
        isOrbitingCameraActive = false;
        mainCamera.enabled = true;
        orbitingCamera.enabled = false;
    }
    
    private void ForceOrbitingCamera()
    {
        isOrbitingCameraActive = true;
        mainCamera.enabled = false;
        orbitingCamera.enabled = true;
    }
} 