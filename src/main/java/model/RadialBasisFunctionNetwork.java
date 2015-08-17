package model;

import lombok.*;
import org.encog.mathutil.rbf.*;
import org.encog.ml.data.*;
import org.encog.neural.rbf.*;
import org.encog.neural.rbf.training.*;

/**
 * Created by ledenev.p on 17.08.2015.
 */

@AllArgsConstructor
@NoArgsConstructor
public class RadialBasisFunctionNetwork implements INetwork {

    private SVDTraining training;

    public static RadialBasisFunctionNetwork create(MLDataSet trainingSet, RBFNetwork network) {
        return new RadialBasisFunctionNetwork(new SVDTraining(network, trainingSet));
    }

    public static RadialBasisFunctionNetwork create(MLDataSet trainingSet, int hiddenSize, RBFEnum type) {

        RBFNetwork network = new RBFNetwork(trainingSet.getInputSize(), hiddenSize, trainingSet.getIdealSize(), type);

        return new RadialBasisFunctionNetwork(new SVDTraining(network, trainingSet));
    }

    public void iteration() {
        training.iteration();
    }

    public double getError() {
        return training.getError();
    }

    public void setThreadCount(int count) {
    }

    public void finishTraining() {
        training.finishTraining();
    }
}
