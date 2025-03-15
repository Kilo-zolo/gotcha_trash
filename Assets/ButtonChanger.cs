using UnityEngine;
using UnityEngine.UI;  // Required to access UI elements

public class ButtonColorChanger : MonoBehaviour
{
    public Image buttonBackground;  // Reference to the background Image component
    public Color clickedColor = Color.green;  // The color you want when clicked
    public Color normalColor = Color.white;  // The color of the background by default

    // Start is called before the first frame update
    void Start()
    {
        // Set the initial color to the normal color
        if (buttonBackground != null)
            buttonBackground.color = normalColor;
    }

    // This method will be called when the button is clicked
    public void OnButtonClicked()
    {
        // Change the background color to the clicked color
        if (buttonBackground != null)
            buttonBackground.color = clickedColor;
    }

    // Optional: Reset to normal color when button is unclicked or any other event
    public void ResetBackgroundColor()
    {
        if (buttonBackground != null)
            buttonBackground.color = normalColor;
    }
}
