package IA;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class filecontrol {

	public void featurewrite(ArrayList<double[]> featureset, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));
			int arraylength = featureset.size();
			int featurelen = featureset.get(0).length;
			for (int i = 0; i < arraylength; i++) {
				for (int j = 0; j < featurelen; j++) {
					out.write(featureset.get(i)[j] + ",");
				}
				out.newLine();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void madatawrite(ArrayList<double[][]> featureset, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));

			for (int i = 0; i < featureset.size(); i++) {
				for (int j = 0; j < featureset.get(i).length; j++) {
					for (int k = 0; k < featureset.get(i)[j].length; k++) {
						out.write(featureset.get(i)[j][k] + ",");
					}
					out.newLine();
				}
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void madatawrite(double[][] featureset, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));
			for (int j = 0; j < featureset.length; j++) {
				for (int k = 0; k < featureset[j].length; k++) {
					out.write(featureset[j][k] + ",");
				}
				out.newLine();
			}
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void datawrite(double[] datax, double[] datay, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));

			for (int j = 0; j < datax.length; j++) {
				out.write(datax[j] + ",");
			}
			out.newLine();
			for (int j = 0; j < datay.length; j++) {
				out.write(datay[j] + ",");
			}
			out.newLine();
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public PPG orippgread(File filepath) {
		PPG ppgs = null;
		ArrayList<Double> ppgx = new ArrayList<Double>();
		ArrayList<Double> ppgy = new ArrayList<Double>();
		ArrayList<Long> ppgtimestamps = new ArrayList<>();

		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			double oldx = 0;
			double oldy = 0;
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				if (tempppg[0].equals("2")) {

					double xvalue = Double.parseDouble(tempppg[1]);
					double yvalue = Double.parseDouble(tempppg[2]);

					if ((xvalue == 0) || (yvalue == 0)) {
						line = in.readLine();
						continue;
					}

					if (ppgx.size() < 1) {

						if (xvalue > 1000000 || yvalue > 1000000 || xvalue < 1000 || yvalue < 1000) {
							line = in.readLine();
							continue;
						} else {
							oldx = xvalue;
							oldy = yvalue;
						}

					}

					double x = Math.abs(xvalue / oldx);
					double y = Math.abs(yvalue / oldy);
//					System.out.print(oldx);
					if (x < 10 && x > 0.1 && y < 10 && y > 0.1) {
						ppgx.add(xvalue);
						ppgy.add(yvalue);
						ppgtimestamps.add(Long.parseLong(tempppg[3]));
						oldx = xvalue;
						oldy = yvalue;
					}
				}
				line = in.readLine();
			}
			in.close();
			Normal_tool normal = new Normal_tool();
			ppgs = new PPG(normal.arraytomatrix(ppgx), normal.arraytomatrix(ppgy),
					normal.arraytomatrix_l(ppgtimestamps));
			normal = null;
		} catch (IOException e) {
			e.printStackTrace();
		}

		return ppgs;
	}

	public Motion orimotionread(File filepath) {

		ArrayList<Double> accx = new ArrayList<Double>();
		ArrayList<Double> accy = new ArrayList<Double>();
		ArrayList<Double> accz = new ArrayList<Double>();

		ArrayList<Double> gyrx = new ArrayList<Double>();
		ArrayList<Double> gyry = new ArrayList<Double>();
		ArrayList<Double> gyrz = new ArrayList<Double>();
		ArrayList<Long> acctimestamps = new ArrayList<>();
		ArrayList<Long> gyrtimestamps = new ArrayList<>();

		Motion motion = null;

		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				double xvalue = Double.parseDouble(tempppg[1]);
				double yvalue = Double.parseDouble(tempppg[2]);
				double zvalue = Double.parseDouble(tempppg[2]);
				if (tempppg[0].equals("0")) {
					accx.add(xvalue);
					accy.add(yvalue);
					accz.add(zvalue);
					acctimestamps.add(Long.parseLong(tempppg[4]));
				}
				if (tempppg[0].equals("1")) {
					gyrx.add(xvalue);
					gyry.add(yvalue);
					gyrz.add(zvalue);
					gyrtimestamps.add(Long.parseLong(tempppg[4]));
				}
				line = in.readLine();
			}
			in.close();
			Normal_tool normal = new Normal_tool();
			motion = new Motion(normal.arraytomatrix(accx), normal.arraytomatrix(accy), normal.arraytomatrix(accz),
					normal.arraytomatrix_l(acctimestamps), normal.arraytomatrix(gyrx), normal.arraytomatrix(gyry),
					normal.arraytomatrix(gyrz), normal.arraytomatrix_l(gyrtimestamps));
			normal = null;
		} catch (IOException e) {
			e.printStackTrace();
		}

		return motion;
	}
}
