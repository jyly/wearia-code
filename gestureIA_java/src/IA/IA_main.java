package IA;

public class IA_main {

	public static dataprocess datapro=new dataprocess();
	public static featureprocess featurepro=new featureprocess();
	public static combine combine=new combine();
	public static void main(String[] args) {

//		�����������ļ���
//		String dirpath = "./testdataset/";
//		360�����������ļ���
		String dirpath = "./oridata/360dataset/";
//		40���û������ļ���
//		String dirpath = "./oridata/40dataset/";
//		���Լ���ͬ���͵�9�����Ƶ��ļ���
		
		// �����Ƶ�����
//		featurepro.all_feature(dirpath);
		//�����Ƶ�ԭʼ���ݶ�
		datapro.all_madata(dirpath);
		// ����Ծ�ֹ״̬�µ�����
//		featurepro.static_feature(dirpath);
		//����Ծ�ֹ״̬�µ�ԭʼ���ݶ�
//		datapro.static_data(dirpath);
		
//		�����Ƶ��ۺ�����
//		combine.all_madata(dirpath);
	}

	
	

}
