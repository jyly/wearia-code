package IA;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class filecontrol {

	public ppg ppgread(File filepath) {
		ppg ppgs = new ppg();
		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				ppgs.x.add(Double.parseDouble(tempppg[0]));
				ppgs.y.add(Double.parseDouble(tempppg[1]));
//				ppgs.timestamps.add(Double.parseDouble(tempppg[2]));
				line = in.readLine();
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ppgs;
	}

	public void ppgwrrite(ppg ppgs) {
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

	public void icawrrite(ppg ppgs) {
		String fileName = "./icappg.csv";
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

	public void featurewrite(ArrayList<feature> featureset, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));
			int arraylength = featureset.size();
			int featurelen = featureset.get(0).features.size();
			for (int i = 0; i < arraylength; i++) {
				for (int j = 0; j < featurelen; j++) {
					out.write(featureset.get(i).features.get(j) + ",");
				}
				out.newLine();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public ppg orisegmentread(File filepath) {
		ppg ppgs = new ppg();

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
					ppgs.timestamps.add(Double.parseDouble(tempppg[3]));
				}
				line = in.readLine();
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return ppgs;
	}

}
