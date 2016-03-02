import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Maxent {

	List<Instance> instanceList = new ArrayList<Instance>();
	List<Feature> featureList = new ArrayList<Feature>();
	List<Integer> featureCountList = new ArrayList<Integer>();
	static List<String> labels = new ArrayList<String>();
	float[] weight = null;
	float[] lastweight = null;
	float[] empiricalE = null;
	float[] modelE = null;
	int M = 0;
	
	public static void main(String[] args) throws IOException {
		if(args.length == 2)
		{
			//String path = "data/train.txt";
			String path = args[0];
			Maxent maxent = new Maxent();
			maxent.loadData(path);
			maxent.train(200);
			//String pathtest = "data/test.txt";

			String pathtest = args[1];
			BufferedReader br = new BufferedReader(new FileReader(new File(pathtest)));
			String line = br.readLine();
			while(line != null) {
				String[] segs = line.split("\t");	
				List<String> fieldList = new ArrayList<String>();
				for(int i =0 ;i< segs.length;i++)
					fieldList.add(segs[i]);

				float[] prob = maxent.predict(fieldList);
				float prob_max;
				int index_maxent = 0;
				
				prob_max = prob[0];
				for(int p=0;p<prob.length;p++)
				{
					if(prob[p]>prob_max)   // 判断最大值
					{
						prob_max=prob[p];
						index_maxent = p;
					}
				}	
				System.out.println(labels.get(index_maxent));
				//System.out.println(Arrays.toString(prob));		
				line = br.readLine();
			}
		}
		
	}
	
	/**
	 * 加载数据，并且将几个变量赋值
	 * featureList：特征函数的list
	 * featureCountList:与特征函数一一对应的，特征函数出现的次数
	 * instanceList:样本数据list
	 * labels:类别list
	 * @param path
	 * @throws IOException
	 */
	public void loadData(String path) throws IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(new File(path)));
		String line = br.readLine();
		while(line != null) {
			String[] segs = line.split("\t");
			String label = segs[0];
			List<String> fieldList = new ArrayList<String>();
			for(int i = 1; i < segs.length; i++) {
				fieldList.add(segs[i]);
				Feature feature = new Feature(label, segs[i]);
				int index = featureList.indexOf(feature);
				if(index == -1){
					featureList.add(feature);
					featureCountList.add(1);
				} else {
					featureCountList.set(index, featureCountList.get(index) + 1);
				}
			}
			if(fieldList.size() > M) M = fieldList.size();
			Instance instance = new Instance(label,fieldList);
			instanceList.add(instance);
			if(labels.indexOf(label) == -1) labels.add(label);
			line = br.readLine();
		}
	}
	
	public void train(int maxIt) {
		
		initParams();
		for(int i = 0; i < maxIt; i++) {
			modeE();
			for(int w = 0; w < weight.length; w++) {
				lastweight[w] = weight[w];
				weight[w] += 1.0 / M * Math.log(empiricalE[w] / modelE[w]);
			}
			//System.out.println(Arrays.toString(weight));
			if(checkExist(lastweight, weight)) break;
		}
	}
	
	public float[] predict(List<String> fieldList) {
		
		float[] prob = calProb(fieldList);
		//System.out.println(labels);
		return prob;
	}
	
	public boolean checkExist(float[] w1, float[] w2) {
		
		for(int i = 0; i < w1.length; i++) {
			if(Math.abs(w1[i] - w2[i]) >= 0.01)
				return false;
		}
		return true;
	}
	
	/**
	 * 初始化一些变量
	 * 计算特征函数的经验期望，就是特征函数出现的次数/样本数
	 */
	public void initParams() {
		
		int size = featureList.size();
		weight = new float[size];
		lastweight = new float[size];
		empiricalE = new float[size];
		modelE = new float[size];
		
		for(int i = 0; i < size; i++) {
			empiricalE[i] = (float)featureCountList.get(i) / instanceList.size();
		}
	}
	
	/**
	 * 计算模型期望，即在当前的特征函数的权重下，计算特征函数的模型期望值。
	 */
	public void modeE() {
		
		modelE = new float[modelE.length];
		for(int i = 0; i < instanceList.size(); i++) 
		{
			List<String> fieldList = instanceList.get(i).fieldList;
			//计算当前样本X对应所有类别的概率
			float[] pro = calProb(fieldList);
			for(int j = 0; j < fieldList.size(); j++) {
				for(int k = 0; k < labels.size(); k++) {
					Feature feature = new Feature(labels.get(k), fieldList.get(j));
					int index = featureList.indexOf(feature);
					if(index != -1)
						modelE[index] += pro[k] * (1.0 / instanceList.size());
				}
			}
		}
	}
	
	//计算p(y|x),此时的x指的是instance里的field
	public float[] calProb(List<String> fieldList) {	
		float[] p = new float[labels.size()];
		float sum = 0;
		for(int i = 0; i < labels.size(); i++) {
			float weightSum = 0;
			for(String field : fieldList) {
				Feature feature = new Feature(labels.get(i), field);
				int index = featureList.indexOf(feature);
				if(index != -1)
					weightSum += weight[index];
			}
			p[i] = (float) Math.exp(weightSum);
			sum += p[i];
		}
		
		for(int i = 0; i < p.length; i++) {
			p[i] /= sum;
		}
		//System.out.println(labels);
		return p;
	}
	
	class Instance {
		
		String label;
		List<String> fieldList = new ArrayList<String>();
		public Instance(String label, List<String> fieldList) {
			this.label = label;
			this.fieldList = fieldList;
		}
	}
	
	class Feature{
		
		String label;
		String value;
		public Feature(String label, String value) {
			this.label = label;
			this.value = value;
		}
		public boolean equals(Object obj) {
			Feature feature = (Feature) obj;
			if(this.label.equals(feature.label) && this.value.equals(feature.value))
				return true;
			return false;
		}
		public String toString() {
			return "[" + label + ", " + value + "]";
		}
		
	}
}
