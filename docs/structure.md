# Structure

Here is the class structure of the code:

```{eval-rst}
.. digraph:: structure
       
   rankdir="LR";
 
   Simulator [href="autoapi/mind_the_gaps/simulator/Simulator.html", target="_top"];
   GPModelling [href="autoapi/mind_the_gaps/gpmodelling/GPModelling.html", target="_top"];
       
   subgraph clusterKernelModels {
      label = "Kernel Models";
        
      Lorentzian [href="autoapi/mind_the_gaps/models/celerite_models/Lorentzian.html", target="_top"];
      Cosinus [href="autoapi/mind_the_gaps/models/celerite_models/Cosinus.html", target="_top"];
      DampedRandomWalk [href="autoapi/mind_the_gaps/models/celerite_models/DampedRandomWalk.html", target="_top"];
      BendingPowerlaw [href="autoapi/mind_the_gaps/models/celerite_models/BendingPowerLaw.html", target="_top"];
      "celerite.terms.Term";
        
      Lorentzian -> "celerite.terms.Term" [arrowhead="empty"];
      Cosinus -> "celerite.terms.Term" [arrowhead="empty"];
      DampedRandomWalk -> "celerite.terms.Term" [arrowhead="empty"];
      BendingPowerlaw -> "celerite.terms.Term" [arrowhead="empty"];
   }
       
   subgraph clusterMeanModels {
      label = "Mean Models";
       
      GaussianModel [href="autoapi/mind_the_gaps/models/celerite_models/GaussianModel.html", target="_top"];
      SineModel [href="autoapi/mind_the_gaps/models/celerite_models/SineModel.html", target="_top"];
      TwoSineModel [href="autoapi/mind_the_gaps/models/celerite_models/TwoSineModel.html", target="_top"];
      LinearModel [href="autoapi/mind_the_gaps/models/celerite_models/LinearModel.html", target="_top"];
      LensingProfileModel [href="autoapi/mind_the_gaps/models/celerite_models/LensingProfileModel.html", target="_top"];
      "celerite.modeling.Model";
        
      LinearModel -> "celerite.modeling.Model" [arrowhead="empty"];
      SineModel -> "celerite.modeling.Model" [arrowhead="empty"];
      TwoSineModel -> "celerite.modeling.Model" [arrowhead="empty"];
      GaussianModel -> "celerite.modeling.Model" [arrowhead="empty"];
      LensingProfileModel -> "celerite.modeling.Model" [arrowhead="empty"]; 
   }
    
   LinearModel -> GPModelling [arrowhead="diamond"];
   GaussianModel -> GPModelling [arrowhead="diamond"];
    
        
   subgraph clusterLightcurve {
      label = "Lightcurves";
    
      GappyLightcurve [href="autoapi/mind_the_gaps/lightcurves/gappylightcurve/GappyLightcurve.html", target="_top"];
      SimpleLightcurve [href="autoapi/mind_the_gaps/lightcurves/simplelightcurve/SimpleLightcurve.html", target="_top"];
      FermiLightcurve [href="autoapi/mind_the_gaps/lightcurves/fermilightcurve/FermiLightcurve.html", target="_top"];
      SwiftLightcurve [href="autoapi/mind_the_gaps/lightcurves/swiftlightcurve/SwiftLightcurve.html", target="_top"];
        
      SimpleLightcurve -> GappyLightcurve [arrowhead="empty"];
      FermiLightcurve -> GappyLightcurve [arrowhead="empty"];
      SwiftLightcurve -> GappyLightcurve [arrowhead="empty"];
   }
        
   subgraph clusterNoise {
      label = "Noise Models";
    
      kraft_pdf [href="autoapi/mind_the_gaps/stats/kraft_pdf.html", target="_top"];    
    
      BaseNoise [href="autoapi/mind_the_gaps/noise_models/BaseNoise.html", target="_top"];
      PoissonNoise [href="autoapi/mind_the_gaps/noise_models/PoissonNoise.html", target="_top"];
      KraftNoise [href="autoapi/mind_the_gaps/noise_models/KraftNoise.html", target="_top"];
      GaussianNoise [href="autoapi/mind_the_gaps/noise_models/GaussianNoise.html", target="_top"];

      kraft_pdf -> KraftNoise [arrowhead="diamond"];
        
      PoissonNoise -> BaseNoise [arrowhead="empty", style="dashed"];
      KraftNoise -> BaseNoise [arrowhead="empty", style="dashed"];
      GaussianNoise -> BaseNoise [arrowhead="empty", style="dashed"];       
    }  
    
   subgraph clusterSimulator {
      label = "Simulator Methods";
    
      BaseSimulatorMethod [href="autoapi/mind_the_gaps/simulator/BaseSimulatorMethod.html", target="_top"];
      TK95Simulator [href="autoapi/mind_the_gaps/simulator/TK95Simulator.html", target="_top"];
      E13Simulator [href="autoapi/mind_the_gaps/simulator/E13Simulator.html", target="_top"];
    
      E13Simulator -> BaseSimulatorMethod [arrowhead="empty", style="dashed"];
      TK95Simulator -> BaseSimulatorMethod [arrowhead="empty", style="dashed"];
   }
    
   BaseSimulatorMethod -> Simulator [arrowhead="diamond"];
   BaseNoise -> Simulator [arrowhead="diamond"];
    
   GappyLightcurve -> Simulator [arrowhead="ediamond"];
   "celerite.terms.Term" -> GPModelling [arrowhead="ediamond"];
   GappyLightcurve -> GPModelling [arrowhead="ediamond"];

```
