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

//		�����������ļ���
//		String dirpath = "./testdataset/";
//		360�����������ļ���
		String dirpath = "./oridata/360dataset/";
//		40���û������ļ���
//		String dirpath = "./oridata/40dataset/";
//		���Լ���ͬ���͵�9�����Ƶ��ļ���
		//		String dirpath = "./oridata/new9dataset/";
		// �����Ƶ�����
		featurepro.all_feature(dirpath);
		//�����Ƶ�ԭʼ���ݶ�
//		datapro.all_madata(dirpath);
		// ����Ծ�ֹ״̬�µ�����
//		featurepro.static_feature(dirpath);
		//����Ծ�ֹ״̬�µ�ԭʼ���ݶ�
//		datapro.static_data(dirpath);
	}

	
	

}
