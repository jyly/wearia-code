package IA;

public class IA_main {

	public static dataprocess datapro=new dataprocess();
	public static featureprocess featurepro=new featureprocess();
	public static combine combine=new combine();
	public static void main(String[] args) {

//		测试用数据文件夹
//		String dirpath = "./testdataset/";
//		360个手势类别的文件夹
		String dirpath = "./oridata/360dataset/";
//		40个用户类别的文件夹
//		String dirpath = "./oridata/40dataset/";
//		我自己不同场和的9个手势的文件夹
		
		// 求手势的特征
//		featurepro.all_feature(dirpath);
		//求手势的原始数据段
		datapro.all_madata(dirpath);
		// 求相对静止状态下的特征
//		featurepro.static_feature(dirpath);
		//求相对静止状态下的原始数据段
//		datapro.static_data(dirpath);
		
//		求手势的综合数据
//		combine.all_madata(dirpath);
	}

	
	

}
