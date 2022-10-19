var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');



var indexRouter = require('./routes/index');
var getBlocknumberRouter = require('./routes/getblocknumber');
var getNftsRouter = require('./routes/getnfts');
var getNftMetadataRouter = require('./routes/getnftmetadata');
var getNftOwnersRouter = require('./routes/getnftowners');
var getSendTransaction = require('./routes/getsendtransaction');
var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/getblocknumber', getBlocknumberRouter);
app.use('/getnfts',getNftsRouter );
app.use('/getmetadata',getNftMetadataRouter );
app.use('/getowners',getNftOwnersRouter );
app.use('/getsendtransaction',getSendTransaction );


// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
