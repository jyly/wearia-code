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

	public Ppg orippgread(File filepath) {
		Ppg ppgs = null;
		ArrayList<Double> ppgx = new ArrayList<Double>();
		ArrayList<Double> ppgy = new ArrayList<Double>();
		ArrayList<Long> ppgtimestamps = new ArrayList<>();

		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			while (line != null) {
//				System.out.println(line);
				String[] tempppg = line.split(",");
				if (tempppg[0].equals("2")) {
					ppgx.add(Double.parseDouble(tempppg[1]));
					ppgy.add(Double.parseDouble(tempppg[2]));
					ppgtimestamps.add(Long.parseLong(tempppg[3]));
				}
				line = in.readLine();
			}
			in.close();
			Normal_tool normal = new Normal_tool();
			ppgs = new Ppg(normal.arraytomatrix(ppgx), normal.arraytomatrix(ppgy),
					normal.arraytomatrix_l(ppgtimestamps));
			normal=null;
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
				if (tempppg[0].equals("0")) {
					accx.add(Double.parseDouble(tempppg[1]));
					accy.add(Double.parseDouble(tempppg[2]));
					accz.add(Double.parseDouble(tempppg[3]));
					acctimestamps.add(Long.parseLong(tempppg[4]));
				}
				if (tempppg[0].equals("1")) {
					gyrx.add(Double.parseDouble(tempppg[1]));
					gyry.add(Double.parseDouble(tempppg[2]));
					gyrz.add(Double.parseDouble(tempppg[3]));
					gyrtimestamps.add(Long.parseLong(tempppg[4]));
				}
				line = in.readLine();
			}
			in.close();
			Normal_tool normal = new Normal_tool();
			motion = new Motion(normal.arraytomatrix(accx), normal.arraytomatrix(accy),normal.arraytomatrix(accz),normal.arraytomatrix_l(acctimestamps),
					normal.arraytomatrix(gyrx), normal.arraytomatrix(gyry),normal.arraytomatrix(gyrz),normal.arraytomatrix_l(gyrtimestamps));
			normal=null;
		} catch (IOException e) {
			e.printStackTrace();
		}

		return motion;
	}
}
