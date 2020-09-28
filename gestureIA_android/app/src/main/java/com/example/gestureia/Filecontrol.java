package com.example.gestureia;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;


public class Filecontrol {

	public void basedfeaturewrite(Context context,ArrayList<float[][]> featureset) {
		int arraylength = featureset.size();
		int featurelen = featureset.get(0)[0].length;
		try {
			String fileName = context.getExternalFilesDir("").getAbsolutePath() + "ppgbasedfeature.csv";//文件存储路径
			Log.e(">>>","filename:"+fileName);
			File file=new File(fileName);
			if(file.exists()){
				file.delete();
				file.createNewFile();
			}
			BufferedWriter out = new BufferedWriter(new FileWriter(file));
			for (int i = 0; i < arraylength; i++) {
				for (int j = 0; j < featurelen; j++) {
					out.write(featureset.get(i)[0][j] + ",");
				}
				out.newLine();
			}
			out.close();

			fileName = context.getExternalFilesDir("").getAbsolutePath() + "motionbasedfeature.csv";//文件存储路径
			Log.e(">>>","filename:"+fileName);
			file=new File(fileName);
			if(file.exists()){
				file.delete();
				file.createNewFile();
			}
			out = new BufferedWriter(new FileWriter(file));
			for (int i = 0; i < arraylength; i++) {
				for (int j = 0; j < featurelen; j++) {
					out.write(featureset.get(i)[1][j] + ",");
				}
				out.newLine();
			}
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
