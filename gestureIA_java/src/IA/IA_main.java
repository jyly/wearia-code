package IA;

public class IA_main {

	// 提取手势行为片段的数据
	public static dataprocess datapro = new dataprocess();
	// 提取手势行为片段的特征
	// public static featureprocess featurepro = new featureprocess();
	// public static combine combine = new combine();

	public static void main(String[] args) {

		// 提取数据的手势片段或手势片段的特征
		String dirpath = "./oridata/";


		// 求手势的特征
		// featurepro.all_feature(dirpath);
		// 求手势的原始数据段
		datapro.all_madata(dirpath);

		// 求相对静止状态下的特征
		// featurepro.static_feature(dirpath);
		// 求相对静止状态下的原始数据段
		// datapro.static_data(dirpath);

		// 求手势的综合数据
		// combine.all_madata(dirpath);
	}

}
