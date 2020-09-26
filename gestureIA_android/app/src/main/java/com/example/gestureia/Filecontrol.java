package com.example.gestureia;

import android.content.Context;

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

	public Ppg ppgread(File filepath) {
		Ppg ppgs = new Ppg();
		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				ppgs.x.add(Double.parseDouble(tempppg[0]));
				ppgs.y.add(Double.parseDouble(tempppg[1]));
				ppgs.timestamps.add(Long.parseLong(tempppg[2]));
				line = in.readLine();
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ppgs;
	}


	public Ppg ppgreadin(InputStream parameterinput) {
		Ppg ppgs = new Ppg();

		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(parameterinput));
			String line = "";
			line = in.readLine();
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				if (tempppg[0].equals("2")) {
					ppgs.x.add(Double.parseDouble(tempppg[1]));
					ppgs.y.add(Double.parseDouble(tempppg[2]));
					ppgs.timestamps.add(Long.parseLong(tempppg[3]));
				}
				line = in.readLine();
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return ppgs;
	}


	public void ppgwrrite(Ppg ppgs) {
		String fileName = "./ppg.csv";
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(fileName));
			int arraylength = ppgs.x.size();
			for (int i = 0; i < arraylength; i++) {
				out.write(ppgs.x.get(i) + "," + ppgs.y.get(i) + ",");
				out.newLine();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public Ppg orisegmentread(File filepath) {
		Ppg ppgs = new Ppg();

		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				if (tempppg[0].equals("2")) {
					ppgs.x.add(Double.parseDouble(tempppg[1]));
					ppgs.y.add(Double.parseDouble(tempppg[2]));
					ppgs.timestamps.add(Long.parseLong(tempppg[3]));
				}
				line = in.readLine();
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return ppgs;
	}


	public void oridatawrite(Context context,Ppg ppgs,Motion motions) {
		long downtime = System.currentTimeMillis();
		String fileName = context.getExternalFilesDir("").getAbsolutePath() + downtime+".csv";//文件存储路径
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(fileName));
			for (int i = 0; i < motions.accx.size(); i++) {
				out.write(0+","+motions.accx.get(i) + "," + motions.accy.get(i) + ","+ motions.accz.get(i) + ","+motions.acctimestamps.get(i));
				out.newLine();
			}
			for (int i = 0; i < motions.gyrx.size(); i++) {
				out.write(1+","+motions.gyrx.get(i) + "," + motions.gyry.get(i) + ","+ motions.gyrz.get(i) + ","+motions.gyrtimestamps.get(i));
				out.newLine();
			}
			for (int i = 0; i < ppgs.x.size(); i++) {
				out.write(2+","+ppgs.x.get(i) + "," + ppgs.y.get(i) + ","+ppgs.timestamps.get(i));
				out.newLine();
			}
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	public void basedfeaturewrite(Context context,ArrayList<float[]> featureset) {
		try {
			String fileName = context.getExternalFilesDir("").getAbsolutePath() + "basedfeature.csv";//文件存储路径
			File file=new File(fileName);
			if(file.exists()){
				file.delete();
				file.createNewFile();
			}
			BufferedWriter out = new BufferedWriter(new FileWriter(file));
			int arraylength = featureset.size();
			int featurelen = featureset.get(0).length;
			for (int i = 0; i < arraylength; i++) {
				for (int j = 0; j < featurelen; j++) {
					out.write(featureset.get(i)[j] + ",");
				}
				out.newLine();
			}
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
