package networks;

import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.neural.pnn.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class PNNBuilder extends NetworkBuilder {

    @Override
    protected String getName() {
        return "PNN";
    }

    @Override
    protected BasicML createNetwork(MLDataSet trainingSet) {
        return new BasicPNN(PNNKernelType.Gaussian, PNNOutputMode.Classification, trainingSet.getInputSize(), trainingSet.getIdealSize());
    }
}
