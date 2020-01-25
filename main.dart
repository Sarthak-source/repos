import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gradient_app_bar/gradient_app_bar.dart';


void main(){
  SystemChrome.setSystemUIOverlayStyle( SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarBrightness: Brightness.light
  ));

  runApp(MyApp());
}


class MyApp extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {

Container cardVeiw(String name,String heading,String subHeading){
  return Container(
    padding: EdgeInsets.all(20.0) ,
     width: 230.0,
     
     child: Card(
       color: Colors.white,
       elevation: 10.0,
       shape: RoundedRectangleBorder(
         borderRadius: BorderRadius.circular(20),
       ),
        child: Wrap(
            children: <Widget>[
              new Container(
                padding: EdgeInsets.all(25.0),
                child:Image.asset(name,
                width: 180.0,
                fit: BoxFit.cover,
                ),
              ),

              ListTile(
               title: Text(heading,
               textAlign: TextAlign.center,
               style: TextStyle(
                 fontFamily: 'VarelaRound',
                 color: Colors.blue[300],
                 fontWeight: FontWeight.bold,
               ),
               ),
               subtitle: Text(subHeading,
               textAlign: TextAlign.center,
               style: TextStyle(
                 fontFamily: 'VarelaRound',
                 color: Colors.blue[300],
               ),),
              ),
            ],
         ),
       ),
    );
}



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: GradientAppBar(
        title: Text('Home',
        style: TextStyle(
          fontFamily: 'VarelaRound',
          fontSize: 20.0,
        )),
    backgroundColorStart: Colors.green[300],
    backgroundColorEnd: Colors.blue[300],
    elevation: 0.0,
    centerTitle: true,
      ),
      body: Center(
          child: Container(
        decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.green[300], Colors.blue[300]])),
        padding: EdgeInsets.all(0.0),
        child: Container(
          alignment: Alignment.topCenter,
          padding: EdgeInsets.only(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: <Widget>[
              Text(
                'comcrop',
                style: TextStyle(
                  fontFamily: 'VarelaRound',
                  fontSize: 35.0,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              Text(
                'commercial crop prices',
                style: TextStyle(
                  fontFamily: 'VarelaRound',
                  fontSize: 15.0,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 70.0),
              Text(
                'Hello there! Tap the card for prices.',
                style: TextStyle(
                  fontFamily: 'VarelaRound',
                  fontSize: 15.0,
                  color: Colors.white,
                ),
              ),
              SizedBox(height: 30.0),
              Container(
                margin: EdgeInsets.symmetric(vertical: 20.0),
                height: 300,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: <Widget>[
                    cardVeiw('assets/ic.png','coffee','prices'), 
                    SizedBox(width: 1),                   
                    cardVeiw('assets/ia.png','arecca','prices'),
                    SizedBox(width: 1), 
                    cardVeiw('assets/ip.png','pepper','prices'),
                    SizedBox(width: 1),  
                  ],
                ),
              ),
            ],
          ),  
        ),
      )));
  }  
} 