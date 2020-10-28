package IA;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

public class IA_main {

	public static dataprocess datapro=new dataprocess();
	public static featureprocess featurepro=new featureprocess();
	public static void main(String[] args) {

//		String dirpath = "./selected_oridata/";
//		String dirpath = "./selected_oridata_2/";
		String dirpath = "./selected_oridata_3/";
//		String dirpath = "./selected_oridata_4/";
		// 求手势的特征
		featurepro.all_feature(dirpath);
//		datapro.all_madata(dirpath);
		// 求相对静止状态下的特征
//		featurepro.static_feature(dirpath);
//		datapro.static_data(dirpath);
	}

	
	

}
