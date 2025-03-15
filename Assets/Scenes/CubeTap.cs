using UnityEngine;
using TMPro; // Import TextMeshPro namespace

public class CubeTap : MonoBehaviour
{
    // Public variable to hold a reference to your TextMeshPro UI element
    public TMP_Text scoreText;

    // Internal score counter
    private int score = 0;

    // This method is called when the cube is tapped or clicked (requires a collider on the cube)
    private void OnMouseDown()
    {
        // Increase score
        score++;
        // Update the UI text
        scoreText.text = "Score: " + score;
    }
}
