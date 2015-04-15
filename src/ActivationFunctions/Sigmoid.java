package ActivationFunctions;

public class Sigmoid extends ActivationFunction {
	private Double gamma;

	public Sigmoid(Double d){
		this.gamma = d;
	}
	
	@Override
	public Double activate(Double d) {
		return 1.0/(1.0 + Math.exp(-gamma*d));
	}

}
