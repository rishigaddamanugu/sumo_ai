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
        // Always create a new dedicated Canvas
        uiCanvas = CreateDedicatedCanvas();

        // Create the speed text UI element
        CreateSpeedTextUI();

        isInitialized = true;
    }

    Canvas CreateDedicatedCanvas()
    {
        GameObject canvasGO = new GameObject("SpeedDisplayCanvas");
        Canvas canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 100; // Ensure it renders above most other UIs

        UnityEngine.UI.CanvasScaler scaler = canvasGO.AddComponent<UnityEngine.UI.CanvasScaler>();
        scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.screenMatchMode = UnityEngine.UI.CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = 0.5f;

        canvasGO.AddComponent<UnityEngine.UI.GraphicRaycaster>();

        Debug.Log("Created dedicated Canvas for SpeedDisplayUI");
        return canvas;
    }

    void CreateSpeedTextUI()
    {
        GameObject speedTextGO = new GameObject("SpeedText");
        speedTextGO.transform.SetParent(uiCanvas.transform, false);

        speedText = speedTextGO.AddComponent<TextMeshProUGUI>();

        TMP_FontAsset defaultFont = Resources.Load<TMP_FontAsset>("Fonts & Materials/LiberationSans SDF");
        if (defaultFont != null)
        {
            speedText.font = defaultFont;
        }
        else
        {
            TMP_FontAsset[] availableFonts = Resources.FindObjectsOfTypeAll<TMP_FontAsset>();
            if (availableFonts.Length > 0)
            {
                speedText.font = availableFonts[0];
            }
        }

        speedText.text = "1x";
        speedText.fontSize = fontSize;
        speedText.color = normalSpeedColor;
        speedText.fontStyle = FontStyles.Bold;
        speedText.alignment = TextAlignmentOptions.BottomRight;
        speedText.enableAutoSizing = false;
        speedText.textWrappingMode = TextWrappingModes.NoWrap;
        speedText.outlineWidth = 0.5f;
        speedText.outlineColor = Color.black;
        speedText.overflowMode = TextOverflowModes.Overflow;

        RectTransform rectTransform = speedText.GetComponent<RectTransform>();
        rectTransform.anchorMin = new Vector2(1f, 0f);
        rectTransform.anchorMax = new Vector2(1f, 0f);
        rectTransform.pivot = new Vector2(1f, 0f);
        rectTransform.anchoredPosition = textOffset;
        rectTransform.sizeDelta = new Vector2(1500f, 600f);

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

        speedText.color = (speedMultiplier > 1f) ? speedupColor : normalSpeedColor;
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

    public void UpdateUISettings(float newFontSize, Color newNormalColor, Color newSpeedupColor, Vector2 newOffset)
    {
        fontSize = newFontSize;
        normalSpeedColor = newNormalColor;
        speedupColor = newSpeedupColor;
        textOffset = newOffset;

        if (speedText != null)
        {
            speedText.fontSize = fontSize;
            speedText.color = normalSpeedColor;

            RectTransform rectTransform = speedText.GetComponent<RectTransform>();
            if (rectTransform != null)
            {
                rectTransform.anchoredPosition = textOffset;
            }
        }
    }

    void OnDestroy()
    {
        if (uiCanvas != null && uiCanvas.name == "SpeedDisplayCanvas")
        {
            int childCount = uiCanvas.transform.childCount;
            if (childCount <= 1)
            {
                DestroyImmediate(uiCanvas.gameObject);
            }
        }
    }
}
