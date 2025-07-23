using UnityEngine;
using TMPro;

public class SpeedDisplayUI : MonoBehaviour
{
    [Header("Display Settings")]
    [SerializeField] private bool showSpeedText = true;
    [SerializeField] private string speedFormat = "{0}x";
    [SerializeField] private Color normalSpeedColor = Color.blue;
    [SerializeField] private Color speedupColor = Color.white;
    [SerializeField] private float fontSize = 400f;
    [SerializeField] private Vector2 textOffset = new Vector2(-50f, 50f);
    
    private TextMeshProUGUI speedText;
    private Canvas uiCanvas;
    private bool isInitialized = false;

    void Start()
    {
        CreateCompleteUISystem();
        HideSpeedText();
    }

    void CreateCompleteUISystem()
    {
        // Create or find the main UI Canvas
        uiCanvas = CreateOrFindCanvas();
        
        // Create the speed text UI element
        CreateSpeedTextUI();
        
        isInitialized = true;
    }

    Canvas CreateOrFindCanvas()
    {
        // First try to find an existing Canvas
        Canvas existingCanvas = FindFirstObjectByType<Canvas>();
        if (existingCanvas != null)
        {
            Debug.Log("Using existing Canvas for speed display");
            return existingCanvas;
        }

        // Create a new Canvas if none exists
        GameObject canvasGO = new GameObject("SpeedDisplayCanvas");
        Canvas canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 100; // Ensure it renders on top
        
        // Add CanvasScaler for proper scaling across different resolutions
        UnityEngine.UI.CanvasScaler scaler = canvasGO.AddComponent<UnityEngine.UI.CanvasScaler>();
        scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.screenMatchMode = UnityEngine.UI.CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = 0.5f;
        
        // Add GraphicRaycaster for UI interactions (if needed)
        canvasGO.AddComponent<UnityEngine.UI.GraphicRaycaster>();
        
        Debug.Log("Created new Canvas for speed display");
        return canvas;
    }

    void CreateSpeedTextUI()
    {
        // Create speed text GameObject
        GameObject speedTextGO = new GameObject("SpeedText");
        speedTextGO.transform.SetParent(uiCanvas.transform, false);
        
        // Add TextMeshProUGUI component
        speedText = speedTextGO.AddComponent<TextMeshProUGUI>();
        
        // Set up the font asset properly to avoid TextCore crashes
        TMP_FontAsset defaultFont = Resources.Load<TMP_FontAsset>("Fonts & Materials/LiberationSans SDF");
        if (defaultFont != null)
        {
            speedText.font = defaultFont;
        }
        else
        {
            // Fallback to any available font
            TMP_FontAsset[] availableFonts = Resources.FindObjectsOfTypeAll<TMP_FontAsset>();
            if (availableFonts.Length > 0)
            {
                speedText.font = availableFonts[0];
            }
        }
        
        // Configure the text component for large, clean display
        speedText.text = "1x";
        speedText.fontSize = fontSize;
        speedText.color = normalSpeedColor;
        speedText.fontStyle = FontStyles.Bold;
        speedText.alignment = TextAlignmentOptions.BottomRight;
        speedText.enableAutoSizing = false;
        speedText.textWrappingMode = TextWrappingModes.NoWrap;
        
        // Add thick outline for better visibility
        speedText.outlineWidth = 0.5f;
        speedText.outlineColor = Color.black;
        
        // Force the text to be visible and not clipped
        speedText.overflowMode = TextOverflowModes.Overflow;
        
        // Set up RectTransform for bottom right corner positioning
        RectTransform rectTransform = speedText.GetComponent<RectTransform>();
        rectTransform.anchorMin = new Vector2(1f, 0f);
        rectTransform.anchorMax = new Vector2(1f, 0f);
        rectTransform.pivot = new Vector2(1f, 0f);
        rectTransform.anchoredPosition = textOffset;
        rectTransform.sizeDelta = new Vector2(1500f, 600f); // Large text bounds
        
        // Initially hide the text
        speedText.enabled = false;
        
        Debug.Log("Speed text UI created successfully");
    }

    // Public methods for external control
    public void ShowSpeedText(float speedMultiplier)
    {
        if (!isInitialized || speedText == null || !showSpeedText) return;
        
        string message = string.Format(speedFormat, speedMultiplier);
        speedText.text = message;
        speedText.enabled = true;
        
        // Change color based on speed
        if (speedMultiplier > 1f)
        {
            speedText.color = speedupColor;
        }
        else
        {
            speedText.color = normalSpeedColor;
        }
    }

    public void HideSpeedText()
    {
        if (speedText != null)
        {
            speedText.enabled = false;
        }
    }

    public void SetSpeedTextVisible(bool visible)
    {
        showSpeedText = visible;
        if (!visible)
        {
            HideSpeedText();
        }
    }

    // Method to update UI settings at runtime
    public void UpdateUISettings(float newFontSize, Color newNormalColor, Color newSpeedupColor, Vector2 newOffset)
    {
        fontSize = newFontSize;
        normalSpeedColor = newNormalColor;
        speedupColor = newSpeedupColor;
        textOffset = newOffset;
        
        if (speedText != null)
        {
            speedText.fontSize = fontSize;
            speedText.color = normalSpeedColor; // Default to normal color
            
            RectTransform rectTransform = speedText.GetComponent<RectTransform>();
            if (rectTransform != null)
            {
                rectTransform.anchoredPosition = textOffset;
            }
        }
    }

    // Cleanup method
    void OnDestroy()
    {
        // If we created the canvas and it's not being used by other components, clean it up
        if (uiCanvas != null && uiCanvas.name == "SpeedDisplayCanvas")
        {
            // Check if there are other UI elements in the canvas
            int childCount = uiCanvas.transform.childCount;
            if (childCount <= 1) // Only our speed text or empty
            {
                DestroyImmediate(uiCanvas.gameObject);
            }
        }
    }
} 