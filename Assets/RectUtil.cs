using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class RectUtil
{
    public static Rect RemoveFromTop(this ref Rect rect, float amount)
    {
        Rect result = new Rect(rect.x, rect.y, rect.width, amount);
        rect.yMin += amount;
        return result;
    }

    public static Rect RemoveFromBottom(this ref Rect rect, float amount)
    {
        Rect result = new Rect(rect.x, rect.yMax - amount, rect.width, amount);
        rect.yMax -= amount;
        return result;
    }

    public static Rect RemoveFromLeft(this ref Rect rect, float amount)
    {
        Rect result = new Rect(rect.x, rect.y, amount, rect.height);
        rect.xMin += amount;
        return result;
    }

    public static Rect RemoveFromRight(this ref Rect rect, float amount)
    {
        Rect result = new Rect(rect.xMax - amount, rect.y, amount, rect.height);
        rect.xMax -= amount;
        return result;
    }

    public static Rect CenteredRect(this Rect outer, Vector2 size)
    {
        float x = Mathf.Round(outer.width - size.x) * 0.5f;
        float y = Mathf.Round(outer.height - size.y) * 0.5f;
        return new Rect(outer.x + x, outer.y + y, size.x, size.y);
    }

    public static Rect Adjusted(this Rect rect, float amount)
    {
        return new Rect(rect.x - amount, rect.y - amount, rect.width + amount * 2, rect.height + amount * 2);
    }

    public static Rect Adjusted(this Rect rect, float x1, float y1, float x2, float y2)
    {
        return new Rect(rect.x - x1, rect.y - y1, rect.width + x1 + x2, rect.height + y1 + y2);
    }
}
