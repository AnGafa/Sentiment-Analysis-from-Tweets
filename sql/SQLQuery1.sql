use SentimentAnalysis
go

drop table dbo.Twitter
drop table dbo.Users

create table Users
(
	UserId uniqueidentifier primary key default newsequentialid(),
	TwitterId numeric(20,0) not null
)

create table Twitter
(
	TweetId int identity(1,1) primary key,
	UserKey uniqueidentifier not null
		constraint twter_userkey_fk references Users(UserId),
	Tweet nvarchar(max) not null
)

If Not Exists(select * from Users where TwitterId=807095)
Begin
insert into Users(TwitterId) values (807095)
End