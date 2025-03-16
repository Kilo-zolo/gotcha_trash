using UnityEngine;
using TMPro;


public class Trashee: MonoBehaviour
{
    private static Trashee instance = new Trashee();
    public string title;
    public static int price = 200;

    public TMP_Text titleText;
    public TMP_Text priceText;

    void Update(){
        priceText.SetText(price.ToString());
        titleText.SetText(title);
    }
}
