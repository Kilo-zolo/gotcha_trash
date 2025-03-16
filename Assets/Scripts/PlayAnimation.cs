using UnityEngine;
using System;
using System.Threading.Tasks;
using static User;
using static Trashee;
using TMPro;

public class PlayAnimation : MonoBehaviour
{
    // Update is called once per frame
    public Animator animator;

    void start(){
        animator = GetComponent<Animator>();
    }
    public void play(){
        
        animator.SetTrigger("play");
        if(User.points > Trashee.price){
            Foo();
        }else{
            TMP_Text text = GameObject.Find("spintext").GetComponent<TMP_Text>();
            text.SetText("Not enough points");
        }
        
    }
    
    public async Task Foo()
    {
        await Task.Delay(2000);
        Animator nextAnimator = GameObject.Find("BinLid").GetComponent<Animator>();
        nextAnimator.SetTrigger("lid");
    }
}
