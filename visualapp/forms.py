from django import forms
from django.contrib.auth.models import User
from django.contrib.auth import authenticate

class signupform(forms.Form):
    Name=forms.CharField(
        label="Username",
        max_length=100,
        widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Username"})
    )
    Firstname = forms.CharField(
        label="Firstname",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "firstname"})
    )
    lastname = forms.CharField(
        label="lastname",
        max_length=100,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "lastname"})
    )
    Email=forms.CharField(
        widget=forms.EmailInput(attrs={"class":"form-control","placeholder":"email"})
    )
    Password=forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class":"form-control","placeholder":"password"})
    )

    

    # def clean(self):
    #     cleaned_data = super().clean()
    #     data = cleaned_data['Mobile']
    #     if data == "abc":
    #         raise forms.ValidationError("mobile must be number")
    def clean_Name(self):
        username = self.cleaned_data.get('Name')
        print(username)
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError('User is already exists')
        return username

    def clean_Password(self):
        data=self.cleaned_data.get('Password')
        print(data)
        if len(data)<8:
            raise forms.ValidationError("password length must be 8")
        return data

    def clean_Email(self):
    # Check that email is not duplicate
        username = self.cleaned_data.get('Name')
        email = self.cleaned_data.get('Email')
        print("y1",email,username)
        users = User.objects.filter(email__iexact=email).exclude(username__iexact=username)
        if users:
            print("y2")
            raise forms.ValidationError("A user with that email already exists.")
        return email.lower()




class loginform(forms.Form):
    Username=forms.CharField(
        label="Username",
        max_length=100,
        widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Username"})
    )
    Password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "password"})
    )
    def clean_Password(self):
        username = self.cleaned_data.get('Username')
        password=self.cleaned_data.get('Password')
        user = authenticate(username=username, password=password)
        print(user)
        if user==None:
            raise forms.ValidationError("username and password did not match")
        return username,password


