package IA;
import java.util.ArrayList;

public class ppg {
	public ArrayList<Double> x = new ArrayList<Double>();
	public ArrayList<Double> y = new ArrayList<Double>();
	public ArrayList<Double> timestamps = new ArrayList<Double>();
	public int fre;
	
	public void noiseremove(int cutspace) {
		
		int index=x.size()-cutspace;
        for (int i =0; i < cutspace; i++) {
            x.remove(index);
            y.remove(index);
            timestamps.remove(index);
        }
    
        for (int i =0; i < cutspace; i++) {
            x.remove(0);
            y.remove(0);
            timestamps.remove(0);
        }
	}
}
