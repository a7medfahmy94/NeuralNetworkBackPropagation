package NeuralNet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNet {
	private Integer numberOfInputNodes;
	private Integer numberOfHiddenNodes;
	private Integer numberOfOutputNodes;
	private ArrayList<Double> hiddenBias;
	private ArrayList<Double> outputBias;
	private ArrayList<ArrayList<Double> > inputHiddenWeights;
	private ArrayList<ArrayList<Double> > hiddenOutputWeights;
		
	public NeuralNet(Integer in,Integer hn,Integer on){
		this.numberOfInputNodes = in;
		this.numberOfHiddenNodes = hn;
		this.numberOfOutputNodes = on;
		
		//initialize input-hidden weights
		this.inputHiddenWeights = new ArrayList<ArrayList<Double>>();
		for(int i = 0 ; i < in; ++i){
			this.inputHiddenWeights.add(new ArrayList<Double>());
			for(int j = 0 ; j < hn; ++j){
				this.inputHiddenWeights.get(i).add(getRandomWeight());
			}

		}

		//initialize hidden-output weights
		this.hiddenOutputWeights = new ArrayList<>();
		for(int i = 0 ; i < hn ; ++i){
			this.hiddenOutputWeights.add(new ArrayList<Double>());
			for(int j = 0 ; j < on ; ++j){
				this.hiddenOutputWeights.get(i).add(getRandomWeight());
			}
		}
		
		//initialize biases
		hiddenBias = new ArrayList<Double>();
		outputBias = new ArrayList<Double>();
		for(int i = 0 ; i < hn ; ++i){
			hiddenBias.add(getRandomWeight());
		}
		for(int i = 0 ; i < on ; ++i){
			outputBias.add(getRandomWeight());
		}
	
	}
	

	public Integer getNumberOfInputNodes(){
		return this.numberOfInputNodes;
	}
	public Integer getNumberOfHiddenNodes(){
		return this.numberOfHiddenNodes;
	}
	public Integer getNumberOfOutputNodes(){
		return this.numberOfOutputNodes;
	}
	public void printNeuralNetwork(){
		System.out.println("=======");
		System.out.println("Input nodes: " + (this.numberOfInputNodes));
		System.out.println("hidden nodes: " + (this.numberOfHiddenNodes));
		System.out.println("output nodes: " + (this.numberOfOutputNodes));
	
		System.out.println("=======");
		for(List<Double> l:this.inputHiddenWeights){
			for(Double d: l){
				printDouble(d);
				System.out.print("    ");
			}
			System.out.println();
		}
		System.out.println("=======");
		
		System.out.println("Hidden Biases");
		for(Double d:hiddenBias){
			System.out.println(d + " ");
		}		
		
		System.out.println("=======");		
		for(List<Double> l:this.hiddenOutputWeights){
			for(Double d: l){
				printDouble(d);
				System.out.print("    ");
			}
			System.out.println();
		}
		System.out.println("=======");
		
		System.out.println("Output Biases");
		for(Double d:outputBias){
			System.out.println(d + " ");
		}		
		System.out.println("=======");

		System.out.println("=======--------=======\n\n");
	}

	public void updateHiddenOutputWeight(Integer i,Integer j,Double d){
		this.hiddenOutputWeights.get(i).set(j, d);
	}
	public void updateInputHiddenWeight(Integer i,Integer j,Double d){
		this.inputHiddenWeights.get(i).set(j, d);
	}
	public void updateHiddenBias(Integer i,Double d){
		this.hiddenBias.set(i, d);
	}
	public void updateOutputBias(Integer i,Double d){
		this.outputBias.set(i, d);
	}
	
	public Double getHiddenBias(Integer i){
		return hiddenBias.get(i);		
	}
	public Double getOutputBias(Integer i){
		return outputBias.get(i);
	}
	public Double getInputHiddenWeight(Integer i,Integer j){
		return this.inputHiddenWeights.get(i).get(j);
	}
	public Double getHiddenOutputWeight(Integer i,Integer j){
		return this.hiddenOutputWeights.get(i).get(j);
	}
	
	/*
	 * Get a random number between [-5.0,5.0]
	 */
	private Double getRandomWeight(){
		Random r = new Random();
		Integer sign = 1;
		if(r.nextDouble() - 0.5 < 0)
			sign = -1;
		return new Random().nextDouble()*5.0*sign;
	}
	private void printDouble(Double d){
		System.out.printf("%.2f" , d);
	}
}
