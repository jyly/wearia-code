package com.example.gestureia;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.PowerManager;

public class Energycontrol {
    private PowerManager.WakeLock wakeLock = null;

    @SuppressLint("InvalidWakeLockTag")
    public void energyopen(Context context) {
        PowerManager pm = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "smartAwake");
        wakeLock.acquire();
    }

    public void energyclose() {
        if (wakeLock != null) {
            wakeLock.release();
            wakeLock = null;
        }
    }

}
